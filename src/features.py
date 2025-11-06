# src/features.py
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Helper engineered features used in many top Titanic solutions
def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    match = re.search(r",\s*([^.]*)\.", str(name))
    return match.group(1).strip() if match else "Unknown"

def add_feature_block(df: pd.DataFrame, add_title=True, add_family=True, add_deck=True) -> pd.DataFrame:
    out = df.copy()
    if add_title and "Name" in out.columns:
        out["Title"] = out["Name"].apply(extract_title)
        # bucket rare titles
        rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
        out["Title"] = out["Title"].replace(
            {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
        )
        out["Title"] = np.where(out["Title"].isin(rare), "Rare", out["Title"])

    if add_family and {"SibSp", "Parch"}.issubset(out.columns):
        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    if add_deck and "Cabin" in out.columns:
        out["Deck"] = out["Cabin"].astype(str).str[0]
        out["Deck"] = out["Deck"].replace({"n": "Unknown", "N": "Unknown", "0": "Unknown"})

    return out

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify numeric & categorical columns automatically
    numeric_features = [c for c in X.columns if X[c].dtype != "object"]
    categorical_features = [c for c in X.columns if X[c].dtype == "object"]

    numeric_transformer = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    categorical_transformer = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]

    # Build the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",   PipelineCompat(numeric_transformer),   numeric_features),
            ("cat",   PipelineCompat(categorical_transformer), categorical_features)
        ]
    )
    return preprocessor

# Minimal stand-in for sklearn.pipeline.Pipeline so we can describe steps as list-of-tuples above
# (Keeps the code explicit and readable for students.)
from sklearn.pipeline import Pipeline
def PipelineCompat(steps):
    return Pipeline(steps=steps)

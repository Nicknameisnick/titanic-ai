# src/data.py
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    # from your problem statement (case-insensitive allowed)
    "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Ticket", "Fare", "Cabin", "Embarked"
]

def load_local_or_uploaded(use_local: bool, train_file_label: str, test_file_label: str):
    if use_local:
        try:
            train_df = pd.read_csv("train.csv")
            test_df = pd.read_csv("test.csv")
            return train_df, test_df
        except Exception as e:
            st.error(f"Could not read local train.csv/test.csv: {e}")
            return None, None
    else:
        train_file = st.sidebar.file_uploader(train_file_label, type=["csv"])
        test_file = st.sidebar.file_uploader(test_file_label, type=["csv"])
        if train_file and test_file:
            return pd.read_csv(train_file), pd.read_csv(test_file)
        return None, None

def basic_clean(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Standard Titanic clean: ensure consistent column name casing
    # (Keep original names if already matched; otherwise patch them)
    def normalize_cols(df):
        # Only normalize known typical Kaggle names if present with different case
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "pclass" and c != "Pclass": rename_map[c] = "Pclass"
            if lc == "sex" and c != "Sex": rename_map[c] = "Sex"
            if lc == "age" and c != "Age": rename_map[c] = "Age"
            if lc == "sibsp" and c != "SibSp": rename_map[c] = "SibSp"
            if lc == "parch" and c != "Parch": rename_map[c] = "Parch"
            if lc == "ticket" and c != "Ticket": rename_map[c] = "Ticket"
            if lc == "fare" and c != "Fare": rename_map[c] = "Fare"
            if lc == "cabin" and c != "Cabin": rename_map[c] = "Cabin"
            if lc == "embarked" and c != "Embarked": rename_map[c] = "Embarked"
            if lc == "survived" and c != "Survived": rename_map[c] = "Survived"
        return df.rename(columns=rename_map)

    train_df = normalize_cols(train_df)
    test_df = normalize_cols(test_df)

    # Embarked: fill missing with mode
    for df in (train_df, test_df):
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    # Fare: fill with median
    for df in (train_df, test_df):
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Age: fill with median (simple baseline; could use group medians)
    for df in (train_df, test_df):
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(df["Age"].median())

    return train_df, test_df

def split_features_target(train_df: pd.DataFrame, target_col: str = "Survived"):
    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])
    # Keep only columns present in test as well to avoid mismatch
    # (assumes test has same columns except Survived)
    return X, y

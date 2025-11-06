# src/modeling.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def get_models(use_lr: bool, use_rf: bool, use_gb: bool, preprocessor):
    models = {}
    if use_lr:
        lr = LogisticRegression(max_iter=1000)
        models["LogisticRegression"] = Pipeline(steps=[("preprocess", preprocessor), ("model", lr)])

    if use_rf:
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=4, random_state=42
        )
        models["RandomForest"] = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])

    if use_gb:
        gb = GradientBoostingClassifier(random_state=42)
        models["GradientBoosting"] = Pipeline(steps=[("preprocess", preprocessor), ("model", gb)])

    return models

def cv_compare_models(models: dict, X, y, cv=5, scoring="accuracy"):
    rows = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring)
        rows.append({
            "model": name,
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": np.array2string(scores, precision=4)
        })
    df = pd.DataFrame(rows).set_index("model").sort_values("mean", ascending=False)
    return df

def fit_on_full_and_predict(best_pipeline, X, y, X_test):
    best_pipeline.fit(X, y)
    preds = best_pipeline.predict(X_test)
    return preds

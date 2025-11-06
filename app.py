# app.py
# Streamlit app for Kaggle Titanic (train -> EDA -> model compare -> submission)
# Built to satisfy both IDS (algorithm, programming, presentation, sources)
# and VA (1D/2D visuals) criteria with extensive comments.

import io
import pandas as pd
import numpy as np
import streamlit as st
from src.data import load_local_or_uploaded, split_features_target, basic_clean
from src.features import build_preprocessor, add_feature_block
from src.modeling import get_models, cv_compare_models, fit_on_full_and_predict
from src.viz import (
    plot_histogram, plot_boxplot, plot_barplot, plot_scatter, palette
)

st.set_page_config(
    page_title="Titanic – ML from Disaster",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic – Machine Learning from Disaster")
st.caption(
    "Template-inspired workflow: EDA → feature engineering → "
    "model comparison (CV) → train on full data → submission.csv"
)

# Sidebar: data
st.sidebar.header("1) Data input")
use_local = st.sidebar.checkbox("Use local train.csv & test.csv (repo root)", value=True)

train_df, test_df = load_local_or_uploaded(
    use_local=use_local,
    train_file_label="Upload train.csv",
    test_file_label="Upload test.csv"
)

if train_df is None or test_df is None:
    st.warning("Please provide both **train.csv** and **test.csv** to continue.")
    st.stop()

st.sidebar.success("Data loaded ✔")

# Show raw data
with st.expander("📄 Preview data (train / test)"):
    st.write("Train shape:", train_df.shape)
    st.dataframe(train_df.head(20), use_container_width=True)
    st.write("Test shape:", test_df.shape)
    st.dataframe(test_df.head(20), use_container_width=True)

# Basic cleaning (standard Titanic handling for Age/Embarked/Fare etc.)
train_df, test_df = basic_clean(train_df.copy(), test_df.copy())

# Add feature engineering (Title/FamilySize/IsAlone/Deck optional)
with st.expander("🧩 Feature engineering"):
    add_title = st.checkbox("Add Title extracted from Name", value=True)
    add_family = st.checkbox("Add FamilySize & IsAlone", value=True)
    add_deck = st.checkbox("Add Cabin Deck (first letter)", value=True)

train_df = add_feature_block(train_df, add_title, add_family, add_deck)
test_df = add_feature_block(test_df, add_title, add_family, add_deck)

# --- EDA Section (VA criteria) ---
st.header("🔎 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Histogram (1D)")
    num_col = st.selectbox(
        "Select a numerical column",
        options=[c for c in train_df.columns if train_df[c].dtype != "object" and c != "Survived"],
        index=0 if "Age" in train_df.columns else 0
    )
    fig_hist = plot_histogram(train_df, num_col, hue="Survived")
    st.pyplot(fig_hist)
    st.caption("Compare distributions per survival class → comment on mean/median differences.")

with col2:
    st.subheader("Boxplot (1D)")
    num_box = st.selectbox(
        "Select a numerical column for boxplot",
        options=[c for c in train_df.columns if train_df[c].dtype != "object" and c != "Survived"],
        index=0
    )
    fig_box = plot_boxplot(train_df, x="Survived", y=num_box)
    st.pyplot(fig_box)
    st.caption("Compare medians and spread between Survived=0 and Survived=1.")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Barplot (2D)")
    cat_col = st.selectbox(
        "Select a categorical column",
        options=[c for c in train_df.columns if train_df[c].dtype == "object"],
        index=0 if "Sex" in train_df.columns else 0
    )
    fig_bar = plot_barplot(train_df, x=cat_col, hue="Survived")
    st.pyplot(fig_bar)
    st.caption("Inspect class balance per category. Decide what to investigate further.")

with col4:
    st.subheader("Scatter (2D)")
    numeric_cols = [c for c in train_df.columns if train_df[c].dtype != "object"]
    if len(numeric_cols) >= 3:
        x_sc = st.selectbox("X axis", options=numeric_cols, index=numeric_cols.index("Age") if "Age" in numeric_cols else 0)
        y_sc = st.selectbox("Y axis", options=[c for c in numeric_cols if c != x_sc], index=0)
        fig_sc = plot_scatter(train_df, x_sc, y_sc, c="Survived")
        st.pyplot(fig_sc)
        st.caption("Check relationships between numeric features (e.g., Age vs Fare).")
    else:
        st.info("Not enough numeric columns for a scatter plot.")

# --- Modeling Section ---
st.header("🤖 Modeling & Evaluation")

# Select features and target
X, y = split_features_target(train_df, target_col="Survived")
X_test = test_df[X.columns]  # keep same feature set

# Preprocessor
preprocessor = build_preprocessor(X)

# Models to compare
st.subheader("Choose models to compare (cross-validation)")
with st.expander("Models"):
    use_lr = st.checkbox("Logistic Regression (baseline)", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    use_gb = st.checkbox("Gradient Boosting", value=True)

models = get_models(use_lr=use_lr, use_rf=use_rf, use_gb=use_gb, preprocessor=preprocessor)

cv_folds = st.slider("CV folds", min_value=3, max_value=10, value=5, step=1)
metric = st.selectbox("CV metric", options=["accuracy"], index=0)
scores_df = cv_compare_models(models, X, y, cv=cv_folds, scoring=metric)

st.write("CV Results:")
st.dataframe(scores_df.style.background_gradient(cmap="Blues"), use_container_width=True)
best_model_name = scores_df.sort_values("mean", ascending=False).index[0]
st.success(f"Best CV mean score: **{best_model_name}** ({scores_df.loc[best_model_name, 'mean']:.4f})")

# --- Train on full data & predict test ---
st.header("📦 Train on full data & create submission")

passenger_id_col = "PassengerId" if "PassengerId" in test_df.columns else st.text_input(
    "Passenger ID column name in test.csv", value="PassengerId"
)

if st.button("Train best model on full train & predict test"):
    best_model = models[best_model_name]
    preds = fit_on_full_and_predict(best_model, X, y, X_test)
    # Kaggle wants 'PassengerId' + 'Survived'
    submission = pd.DataFrame({
        passenger_id_col: test_df[passenger_id_col].values,
        "Survived": preds.astype(int)
    })
    st.dataframe(submission.head(20), use_container_width=True)
    # Download button
    buf = io.BytesIO()
    submission.to_csv(buf, index=False)
    st.download_button(
        label="⬇️ Download submission.csv",
        data=buf.getvalue(),
        file_name="submission.csv",
        mime="text/csv"
    )
    st.success("Submission file generated. Upload it to Kaggle!")

# --- Documentation / Reflection (IDS) ---
with st.expander("ℹ️ Explanation & choices (for presentation criteria)"):
    st.markdown("""
**Why these algorithms?**
- *Logistic Regression*: strong baseline, interpretable coefficients.
- *Random Forest*: handles non-linearity & interactions, robust to outliers.
- *Gradient Boosting*: often strong performance on tabular competitions.

**Evaluation method (train set)**
- We use **Stratified K-Fold CV** with accuracy to compare ideas fairly (meets IDS ‘Uitstekend’).

**Feature engineering choices**
- `Title` (from name), `FamilySize` and `IsAlone`, `Deck` (from cabin) are classic Titanic features with known predictive power.

**np.where & datetime (quick reference)**
- `np.where(cond, a, b)` → vectorized if–else. Example: `df['IsChild'] = np.where(df['Age'] < 12, 1, 0)`
- `pd.to_datetime(df['col'])` + `.dt.year/.month/.day` to extract date parts. Format parse: `pd.to_datetime(df['col'], format="%Y-%m-%d")`
""")

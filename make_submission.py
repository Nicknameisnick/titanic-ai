# make_submission.py
# Train the best model (by CV accuracy) on full train and create submission.csv
import argparse
import pandas as pd
from src.data import basic_clean, split_features_target
from src.features import add_feature_block, build_preprocessor
from src.modeling import get_models, cv_compare_models, fit_on_full_and_predict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="train.csv")
    parser.add_argument("--test_path", default="test.csv")
    parser.add_argument("--passenger_id_col", default="PassengerId")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--no_title", action="store_true", help="Disable Title feature")
    parser.add_argument("--no_family", action="store_true", help="Disable FamilySize/IsAlone")
    parser.add_argument("--no_deck", action="store_true", help="Disable Deck feature")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    train_df, test_df = basic_clean(train_df, test_df)

    train_df = add_feature_block(train_df, add_title=not args.no_title, add_family=not args.no_family, add_deck=not args.no_deck)
    test_df  = add_feature_block(test_df,  add_title=not args.no_title, add_family=not args.no_family, add_deck=not args.no_deck)

    X, y = split_features_target(train_df, target_col="Survived")
    X_test = test_df[X.columns]

    preprocessor = build_preprocessor(X)
    models = get_models(use_lr=True, use_rf=True, use_gb=True, preprocessor=preprocessor)
    cv_results = cv_compare_models(models, X, y, cv=args.cv, scoring="accuracy")
    best_name = cv_results.sort_values("mean", ascending=False).index[0]
    best_model = models[best_name]

    preds = fit_on_full_and_predict(best_model, X, y, X_test)
    submission = pd.DataFrame({args.passenger_id_col: test_df[args.passenger_id_col], "Survived": preds.astype(int)})
    submission.to_csv(args.out, index=False)
    print(f"Best model: {best_name} (CV mean={cv_results.loc[best_name,'mean']:.4f})")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()



"""Train and compare six classifiers on the cleaned P2P loan dataset.

Pipeline:
  1. Load cleaned CSV.
  2. Stratified train/test split with class oversampling on the train fold.
  3. Fit each model, pick top-15 features by built-in importance, refit on
     those features only (the "hybrid" selection step).
  4. Evaluate, dump plots under reports/figures/, persist the model under models/.
  5. Write a combined evaluation_metrics.csv table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import report, save_metrics_table  # noqa: E402
from src.utils import (  # noqa: E402
    balance_classes,
    ensure_dir,
    load_dataset,
    save_model,
    stratified_split,
    top_n_features,
)

RANDOM_STATE = 42
TOP_K = 15

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "LightGBM": lambda: LGBMClassifier(
        n_estimators=400, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "CatBoost": lambda: CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=6, verbose=False, random_seed=RANDOM_STATE
    ),
}


def get_importances(model, feature_names: list[str]) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).ravel()
    return np.ones(len(feature_names))


def train_one(name: str, factory, X_train, y_train, X_test, y_test):
    print(f"\n=== {name} ===")
    model = factory()
    model.fit(X_train, y_train)

    importances = get_importances(model, list(X_train.columns))
    selected = top_n_features(importances, X_train.columns, n=TOP_K)
    print(f"Top {TOP_K} features selected: {selected}")

    model_refit = factory()
    model_refit.fit(X_train[selected], y_train)

    y_pred = model_refit.predict(X_test[selected])
    y_prob = model_refit.predict_proba(X_test[selected])[:, 1]

    refit_importances = get_importances(model_refit, selected)
    res = report(
        y_test, y_pred, y_prob,
        model_name=name,
        feature_names=selected,
        importances=refit_importances,
    )

    save_model(model_refit, ROOT / "models" / f"{name.lower()}.joblib")
    pd.Series(selected, name="feature").to_csv(
        ROOT / "models" / f"{name.lower()}_features.csv", index=False
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/cleaned_p2p.csv")
    parser.add_argument("--target", default="Defaulted")
    parser.add_argument("--only", nargs="*", choices=list(MODELS), help="Train only these models")
    args = parser.parse_args()

    ensure_dir(ROOT / "models")
    X, y = load_dataset(args.data, target_col=args.target)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=RANDOM_STATE)
    X_train, y_train = balance_classes(X_train, y_train, method="oversample", random_state=RANDOM_STATE)
    print(f"Train shape after balancing: {X_train.shape}, Test shape: {X_test.shape}")

    selected_models = args.only or list(MODELS)
    results = [train_one(name, MODELS[name], X_train, y_train, X_test, y_test) for name in selected_models]
    csv_path = save_metrics_table(results)
    print(f"\nMetrics table written to: {csv_path}")


if __name__ == "__main__":
    main()

"""SHAP-based feature ranking for a trained classifier.

Used alongside built-in feature_importances_ to do hybrid feature selection
(SHAP ∪ model importance) as described in the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_dataset, load_model  # noqa: E402


def shap_top_features(model, X: pd.DataFrame, n: int = 15, sample: int = 2000) -> list[str]:
    if len(X) > sample:
        X = X.sample(sample, random_state=42)
    explainer = shap.Explainer(model, X)
    values = explainer(X, check_additivity=False)
    mean_abs = np.abs(values.values).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)
    s = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    return s.head(n).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a joblib model")
    parser.add_argument("--data", default="data/processed/cleaned_p2p.csv")
    parser.add_argument("--target", default="Defaulted")
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()

    model = load_model(args.model)
    X, _ = load_dataset(args.data, target_col=args.target)
    top = shap_top_features(model, X, n=args.n)
    print("\n".join(top))


if __name__ == "__main__":
    main()

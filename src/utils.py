"""Shared helpers for the credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str | Path, target_col: str = "Defaulted") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    if target_col not in df.columns:
        raise ValueError(f"Target column `{target_col}` not found in {path}")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def balance_classes(
    X: pd.DataFrame, y: pd.Series, method: str = "oversample", random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Quick binary resampler — no extra dependency on imbalanced-learn.

    No-ops if the dataset already has at most one class or is exactly balanced.
    """
    counts = y.value_counts()
    if len(counts) < 2 or counts.iloc[0] == counts.iloc[-1]:
        return X.reset_index(drop=True), y.reset_index(drop=True).astype(int)

    df = X.copy()
    df["__y"] = y.values
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    majority = df[df["__y"] == majority_label]
    minority = df[df["__y"] == minority_label]

    if method == "oversample":
        minority_up = minority.sample(len(majority), replace=True, random_state=random_state)
        out = pd.concat([majority, minority_up])
    elif method == "undersample":
        majority_down = majority.sample(len(minority), replace=False, random_state=random_state)
        out = pd.concat([majority_down, minority])
    else:
        raise ValueError("method must be 'oversample' or 'undersample'")

    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out.drop(columns="__y"), out["__y"].astype(int)


def top_n_features(importances: Iterable[float], feature_names: Iterable[str], n: int = 15) -> list[str]:
    s = pd.Series(list(importances), index=list(feature_names))
    return s.sort_values(ascending=False).head(n).index.tolist()


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

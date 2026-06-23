"""Preprocessing pipeline for the Bondora P2P lending dataset.

Loads the raw CSV, drops leaky/ID columns, fills missing values, encodes
categoricals, winsorises numeric columns, and drops highly-correlated
features. The cleaned frame is written to disk and returned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

LEAKY_COLUMNS = ["UserName", "LoanNumber", "PrincipalBalance", "Status"]
CORR_THRESHOLD = 0.95
WINSOR_LOW, WINSOR_HIGH = 0.001, 0.999


def preprocess_data(
    file_path: str | Path,
    save_path: str | Path = "data/processed/cleaned_p2p.csv",
    target_col: str = "Defaulted",
) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset loaded with shape: {df.shape}")

    df.drop(columns=LEAKY_COLUMNS, inplace=True, errors="ignore")
    print(f"Dropped leaky columns: {LEAKY_COLUMNS}")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        print(f"Dropped fully-empty columns: {empty_cols}")

    for col in df.columns:
        if df[col].dtype == "object":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
        elif col != target_col:
            df[col] = df[col].fillna(df[col].mean())
    print("Missing values handled")

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    print("Categorical features encoded")

    if target_col in df.columns:
        print(f"Target `{target_col}` classes: {sorted(df[target_col].unique())}")

    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col == target_col:
            continue
        lower = df[col].quantile(WINSOR_LOW)
        upper = df[col].quantile(WINSOR_HIGH)
        df[col] = df[col].clip(lower, upper)
    print("Winsorisation applied")

    cor_matrix = df.corr().abs()
    upper_triangle = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )
    to_drop_corr = [
        c for c in upper_triangle.columns if any(upper_triangle[c] > CORR_THRESHOLD)
    ]
    df.drop(columns=to_drop_corr, inplace=True)
    print(f"Dropped {len(to_drop_corr)} high-correlation columns")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved to: {save_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/trainset.csv")
    parser.add_argument("--output", default="data/processed/cleaned_p2p.csv")
    parser.add_argument("--target", default="Defaulted")
    args = parser.parse_args()
    preprocess_data(args.input, args.output, args.target)


if __name__ == "__main__":
    main()

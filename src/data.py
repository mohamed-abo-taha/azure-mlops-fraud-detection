"""Load an imbalanced fraud dataset.

Tries the OpenML credit-card fraud dataset (the ULB benchmark, ~0.17% fraud).
If it is unavailable, falls back to a synthetic imbalanced generator so the
repo always runs. The result is cached to data/fraud.parquet.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

TARGET = "Class"


def _synthetic(n=60000, fraud_rate=0.004, seed=0):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=8, n_redundant=4,
        weights=[1 - fraud_rate], flip_y=0.001, class_sep=1.3, random_state=seed,
    )
    cols = [f"V{i}" for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=cols)
    rng = np.random.default_rng(seed)
    df["Amount"] = np.abs(rng.normal(80, 120, size=n))
    df[TARGET] = y.astype(int)
    return df, "synthetic"


def load_fraud(prefer_openml=True, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "fraud.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache), "cache"

    if prefer_openml:
        try:
            from sklearn.datasets import fetch_openml

            ds = fetch_openml(data_id=1597, as_frame=True, parser="auto")
            df = ds.data.copy()
            df[TARGET] = ds.target.astype(int)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df[TARGET] = df[TARGET].astype(int)
            df.to_parquet(cache)
            return df, "openml:creditcard"
        except Exception as e:  # noqa: BLE001
            print("OpenML fetch failed, using synthetic data:", e)

    df, src = _synthetic()
    df.to_parquet(cache)
    return df, src

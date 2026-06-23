"""Scoring model: a calibrated classifier plus a cost-based decision threshold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cost_threshold(y_true, proba, cost_fp=1.0, cost_fn=50.0):
    """Pick the probability threshold that minimises expected cost.

    Missing a fraud (false negative) is treated as far more expensive than a
    false alarm (false positive), so the threshold is pushed below 0.5.
    Returns (threshold, total_cost_at_threshold).
    """
    y_true = np.asarray(y_true)
    candidates = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, 200)))
    best_t, best_cost = 0.5, float("inf")
    for t in candidates:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost


@dataclass
class Scorer:
    model: object
    threshold: float
    feature_names: list
    version: str = "unknown"

    def score_one(self, features: dict) -> dict:
        x = np.array([[float(features.get(f, 0.0)) for f in self.feature_names]])
        p = float(self.model.predict_proba(x)[0, 1])
        return {
            "probability": p,
            "is_fraud": bool(p >= self.threshold),
            "threshold": self.threshold,
            "model_version": self.version,
        }

"""Train the fraud model end to end.

Fit a gradient-boosted classifier with class balancing, calibrate its
probabilities (isotonic, on a held-out split), pick a decision threshold from a
cost model, evaluate on a separate test split, and register the model + metadata
to the configured storage backend (local or Azure Blob).
"""
from __future__ import annotations

import argparse
import json
import os

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from .config import settings
from .data import TARGET, load_fraud
from .evaluate import evaluate, plot_calibration, plot_confusion, plot_pr, recall_at_precision
from .model import cost_threshold
from .registry import ModelRegistry
from .storage import get_storage


def _calibrate(base, X_cal, y_cal):
    # Platt/sigmoid calibration: monotonic, so it preserves ranking (PR-AUC)
    # while improving probability quality. Isotonic was measured to hurt PR-AUC
    # here by collapsing scores into ties on the few positives.
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except Exception:  # noqa: BLE001  (older sklearn)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-openml", action="store_true", help="skip OpenML, use synthetic data")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    df, src = load_fraud(prefer_openml=not args.no_openml)
    print(f"dataset={src} shape={df.shape} fraud_rate={df[TARGET].mean():.5f}", flush=True)
    features = [c for c in df.columns if c != TARGET]
    X = df[features].values.astype("float64")
    y = df[TARGET].values.astype(int)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)

    # Imbalance is handled downstream by isotonic calibration + the cost-based
    # threshold, so the booster is left to fit a strong ranker (class weighting
    # was measured to hurt PR-AUC and top-of-ranking precision here).
    base = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=0,
    )
    base.fit(X_tr, y_tr)
    cal = _calibrate(base, X_cal, y_cal)

    thr, _ = cost_threshold(y_cal, cal.predict_proba(X_cal)[:, 1], settings.cost_fp, settings.cost_fn)
    proba_te = cal.predict_proba(X_te)[:, 1]

    metrics = evaluate(y_te, proba_te, thr)
    metrics["recall_at_p90"] = recall_at_precision(y_te, proba_te, 0.9)
    metrics["pr_auc_uncalibrated"] = float(average_precision_score(y_te, base.predict_proba(X_te)[:, 1]))
    metrics["dataset"] = src
    metrics["fraud_rate"] = float(y.mean())
    metrics["cost_fp"] = settings.cost_fp
    metrics["cost_fn"] = settings.cost_fn

    registry = ModelRegistry(get_storage(settings))
    version = registry.save(cal, thr, features, metrics, version=args.version)
    metrics["version"] = version
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)

    plot_pr(y_te, proba_te, "assets/pr_curve.png")
    plot_calibration(y_te, proba_te, "assets/calibration.png")
    plot_confusion(y_te, (proba_te >= thr).astype(int), "assets/confusion_matrix.png")

    print(
        f"registered {version} | PR-AUC={metrics['pr_auc']:.4f} "
        f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
        f"thr={thr:.4f} recall@p90={metrics['recall_at_p90']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

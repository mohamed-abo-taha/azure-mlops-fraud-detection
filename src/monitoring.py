"""Data-drift detection (PSI + KS) and optional App Insights telemetry."""
from __future__ import annotations

import numpy as np


def psi(expected, actual, bins=10) -> float:
    """Population Stability Index between two samples of one feature."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=edges)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref_df, cur_df, features, psi_threshold=0.2) -> dict:
    from scipy.stats import ks_2samp

    rows = []
    for f in features:
        p = psi(ref_df[f].values, cur_df[f].values)
        ks = float(ks_2samp(ref_df[f].values, cur_df[f].values).statistic)
        rows.append({"feature": f, "psi": round(p, 4), "ks": round(ks, 4), "drift": bool(p > psi_threshold)})
    n_drift = sum(r["drift"] for r in rows)
    return {
        "n_features": len(features),
        "n_drifted": n_drift,
        "overall_drift": bool(n_drift > 0),
        "by_feature": rows,
    }


def setup_app_insights(connection_string: str) -> bool:
    """Wire up Azure Monitor / App Insights if a connection string is provided."""
    if not connection_string:
        return False
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=connection_string)
        return True
    except Exception as e:  # noqa: BLE001
        print("App Insights setup skipped:", e)
        return False

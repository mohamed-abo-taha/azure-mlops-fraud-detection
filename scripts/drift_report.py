"""Generate a data-drift report between a reference sample and a current sample.

Splits the dataset in half and compares the two halves as a stand-in for
"training reference" vs "recent production traffic". In a real deployment the
current sample would be the latest scored batch pulled from storage.
"""
from __future__ import annotations

import argparse
import json
import os

from src.data import TARGET, load_fraud
from src.monitoring import drift_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/drift_report.json")
    args = ap.parse_args()

    df, _ = load_fraud()
    features = [c for c in df.columns if c != TARGET]
    half = len(df) // 2
    report = drift_report(df.iloc[:half], df.iloc[half:], features)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"overall_drift={report['overall_drift']} drifted={report['n_drifted']}/{report['n_features']}")


if __name__ == "__main__":
    main()

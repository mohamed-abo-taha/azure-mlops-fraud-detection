"""A small versioned model registry over a StorageBackend.

Each version stores the serialized model and a metadata JSON (threshold,
feature names, metrics). A `latest` pointer tracks the newest version. Works
identically on local disk or Azure Blob.
"""
from __future__ import annotations

import io
import json
import time

import joblib

from .model import Scorer


def default_version() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class ModelRegistry:
    def __init__(self, storage, name: str = "fraud"):
        self.s = storage
        self.name = name

    def save(self, model, threshold, feature_names, metrics, version=None) -> str:
        version = version or default_version()
        prefix = f"{self.name}/{version}"
        buf = io.BytesIO()
        joblib.dump(model, buf)
        self.s.put_bytes(f"{prefix}/model.joblib", buf.getvalue())
        meta = {
            "version": version,
            "threshold": float(threshold),
            "feature_names": list(feature_names),
            "metrics": metrics,
        }
        self.s.put_bytes(f"{prefix}/metadata.json", json.dumps(meta, indent=2).encode())
        self.s.put_bytes(f"{self.name}/latest.txt", version.encode())
        return version

    def latest_version(self):
        key = f"{self.name}/latest.txt"
        return self.s.get_bytes(key).decode().strip() if self.s.exists(key) else None

    def load(self, version=None) -> Scorer:
        version = version or self.latest_version()
        if not version:
            raise FileNotFoundError("no model registered")
        prefix = f"{self.name}/{version}"
        model = joblib.load(io.BytesIO(self.s.get_bytes(f"{prefix}/model.joblib")))
        meta = json.loads(self.s.get_bytes(f"{prefix}/metadata.json").decode())
        return Scorer(model=model, threshold=meta["threshold"],
                      feature_names=meta["feature_names"], version=version)

"""Runtime settings, read from environment (with a .env fallback)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")  # local | azure_blob
    azure_storage_connection_string: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    blob_container: str = os.getenv("MODEL_BLOB_CONTAINER", "models")
    local_registry_dir: str = os.getenv("LOCAL_REGISTRY_DIR", "artifacts/registry")
    appinsights_conn: str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
    cost_fp: float = float(os.getenv("COST_FP", "1"))
    cost_fn: float = float(os.getenv("COST_FN", "50"))


settings = Settings()

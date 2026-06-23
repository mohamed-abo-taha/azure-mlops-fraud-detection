"""FastAPI scoring service.

Loads the latest model from the registry (local folder or Azure Blob, by
config) and scores transactions. Designed to run as a container on Azure
Container Apps, pulling the model from Blob at startup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .monitoring import setup_app_insights
from .registry import ModelRegistry
from .storage import get_storage

_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = ModelRegistry(get_storage(settings)).load()
    return _scorer


@asynccontextmanager
async def lifespan(app):
    setup_app_insights(settings.appinsights_conn)
    yield


app = FastAPI(title="Fraud scoring service", version="1.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    features: dict


class ScoreResponse(BaseModel):
    probability: float
    is_fraud: bool
    threshold: float
    model_version: str


@app.get("/health")
def health():
    try:
        version = ModelRegistry(get_storage(settings)).latest_version()
    except Exception:  # noqa: BLE001
        version = None
    return {"status": "ok", "model_version": version}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        scorer = _get_scorer()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not available: {e}")
    return scorer.score_one(req.features)

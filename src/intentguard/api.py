"""Local HTTP adapter. One model call at a time; no request text is logged or retained."""

import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from .predictor import Predictor


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: Annotated[
        list[Annotated[StrictStr, Field(min_length=1, max_length=500)]],
        Field(min_length=1, max_length=32),
    ]
    top_k: Annotated[int, Field(strict=True, ge=1, le=5)] = 3

    @field_validator("texts")
    @classmethod
    def nonblank(cls, values):
        if any(not text.strip() for text in values):
            raise ValueError("Blank texts are not supported")
        return values


def create_app(bundle=None, *, predictor=None):
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        engine = predictor if predictor is not None else Predictor(bundle)
        engine.warmup()
        app.state.predictor = engine
        yield

    app = FastAPI(
        title="IntentGuard JAX",
        version="0.1.0",
        lifespan=lifespan,
        description="Intent routing with calibrated confidence and explicit abstention",
    )

    @app.get("/health")
    def health():
        return {"status": "ready", "bundle_id": app.state.predictor.bundle_id}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Inference busy; retry later")
        try:
            return app.state.predictor.predict(request.texts, request.top_k)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            lock.release()

    return app

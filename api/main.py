"""
REST API for the sentiment model.

    uvicorn api.main:app --port 8000
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"texts": ["I love this", "worst day ever"]}'

Interactive docs at /docs. The model is loaded once at startup from
MODEL_SOURCE (a local folder or a Hugging Face repo id); see sentiment/model.py.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from sentiment.model import SentimentModel

log = logging.getLogger("sentiment.api")
MAX_TEXTS = 100
MAX_CHARS = 1000


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS, examples=[["I love this phone", "worst service ever"]])


class Prediction(BaseModel):
    text: str
    sentiment: str = Field(..., examples=["positive"])
    confidence: float = Field(..., ge=0.5, le=1.0, examples=[0.97])


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set app.state.model to a stand-in before startup.
    if getattr(app.state, "model", None) is None:
        started = time.time()
        app.state.model = SentimentModel(threads=int(os.environ.get("TORCH_THREADS", "2")))
        log.info("Loaded model from %s in %.1fs", app.state.model.source, time.time() - started)
    yield


app = FastAPI(
    title="Tweet Sentiment API",
    description="DistilBERT fine-tuned on Sentiment140. Classifies short English text as positive or negative.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.model.source}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    started = time.perf_counter()
    # Very long inputs are truncated: tweets are short and the model reads 48 tokens.
    texts = [t[:MAX_CHARS] for t in body.texts]
    predictions = request.app.state.model.classify(texts)
    return {
        "predictions": predictions,
        "model": request.app.state.model.source,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }

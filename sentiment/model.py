"""
Loads the fine-tuned model and predicts labels with confidences, in batches,
on CPU. Used by the evaluation and inference scripts, the API and the demo.

The model can come from a local folder (models/final after training) or from
the Hugging Face Hub (a repo id such as "ngugiauraa/distilbert-sentiment140").
"""

import os

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from .text import clean_tweet

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MODEL_DIR = os.path.join(ROOT, "models", "final")
HUB_MODEL_ID = "ngugiauraa/distilbert-sentiment140"
BASE_MODEL = "distilbert-base-uncased"
LABEL_NAMES = {0: "negative", 1: "positive"}
MAX_LENGTH = 48  # must match scripts/train.py


def default_source() -> str:
    """MODEL_SOURCE env var, else the local trained model, else the Hub copy."""
    return os.environ.get("MODEL_SOURCE") or (LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HUB_MODEL_ID)


class SentimentModel:
    def __init__(self, source: str | None = None, threads: int = 4):
        self.source = source or default_source()
        torch.set_num_threads(threads)
        # Training checkpoints hold weights only; fine-tuning never changes the
        # tokenizer, so the base model's tokenizer is the right one for them.
        is_local = os.path.isdir(self.source)
        has_tokenizer = not is_local or any(
            os.path.exists(os.path.join(self.source, f)) for f in ("tokenizer.json", "vocab.txt")
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.source if has_tokenizer else BASE_MODEL)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.source)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, texts, batch_size=64, on_batch=None):
        """Returns (labels, confidences) as numpy arrays. Confidence is the
        softmax probability of the predicted label, between 0.5 and 1.0."""
        labels, confidences = [], []
        for start in range(0, len(texts), batch_size):
            batch = [clean_tweet(t) for t in texts[start:start + batch_size]]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            probs = torch.softmax(self.model(**encoded).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            labels.append(pred.numpy())
            confidences.append(conf.numpy())
            if on_batch:
                on_batch(min(start + batch_size, len(texts)), len(texts))
        if not labels:
            return np.array([], dtype=int), np.array([], dtype=float)
        return np.concatenate(labels), np.concatenate(confidences)

    def classify(self, texts):
        """Friendly form for the API and demo: one dict per text."""
        labels, confidences = self.predict(list(texts))
        return [
            {"text": text, "sentiment": LABEL_NAMES[int(label)], "confidence": round(float(conf), 4)}
            for text, label, conf in zip(texts, labels, confidences, strict=True)
        ]

"""
Shared helpers for evaluate.py and inference.py: load a saved model and
predict labels + confidences for a list of texts in batches on CPU.
"""

import os
import re

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODEL_DIR = os.path.join(ROOT, "models", "final")
LABEL_NAMES = {0: "negative", 1: "positive"}
MAX_LENGTH = 48  # must match train.py
BASE_MODEL = "distilbert-base-uncased"


def clean_tweet(text):
    """Same cleaning as preprocess.py, so new text looks like the training data."""
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_dir=DEFAULT_MODEL_DIR, threads=4):
    if not os.path.isdir(model_dir):
        raise SystemExit(f"No model found at {model_dir}. Train first (python scripts/train.py) or pass --model.")
    torch.set_num_threads(threads)
    # Training checkpoints hold weights only. Fine-tuning never changes the
    # tokenizer, so the base model's tokenizer is the right one for them.
    has_tokenizer = os.path.exists(os.path.join(model_dir, "tokenizer.json")) or os.path.exists(os.path.join(model_dir, "vocab.txt"))
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir if has_tokenizer else BASE_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def predict(texts, tokenizer, model, batch_size=64, on_batch=None):
    """Returns (labels, confidences) as numpy arrays. Confidence is the softmax
    probability of the predicted label."""
    labels, confidences = [], []
    for start in range(0, len(texts), batch_size):
        batch = [clean_tweet(t) for t in texts[start:start + batch_size]]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        probs = torch.softmax(model(**encoded).logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        labels.append(pred.numpy())
        confidences.append(conf.numpy())
        if on_batch:
            on_batch(min(start + batch_size, len(texts)), len(texts))
    if not labels:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(labels), np.concatenate(confidences)

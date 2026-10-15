"""
Publishes the trained model and the demo to Hugging Face (both free).

One-time setup:
    1. Create an account at https://huggingface.co
    2. Settings > Access Tokens > New token, type "Write"
    3. huggingface-cli login     (paste the token)

Then, from the project root:
    python scripts/publish_model.py            # model + demo
    python scripts/publish_model.py --model-only

The model card is written from reports/metrics.json, so the published numbers
are always the ones evaluate.py measured.
"""

import argparse
import json
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # run from anywhere

from sentiment.model import HUB_MODEL_ID, LOCAL_MODEL_DIR, ROOT  # noqa: E402

SPACE_ID = "ngugiauraa/tweet-sentiment"


def model_card(metrics: dict) -> str:
    cm = metrics["confusion_matrix"]["rows_actual_cols_predicted"]
    m = metrics
    row = (
        f"| {m['test_rows']:,} | {m['accuracy']:.2%} | {m['f1_macro']:.3f} "
        f"| {m['precision_positive']:.3f} | {m['recall_positive']:.3f} |"
    )
    return f"""---
language: en
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert-base-uncased
datasets:
- sentiment140
metrics:
- accuracy
- f1
---

# DistilBERT fine-tuned on Sentiment140

Binary sentiment (negative / positive) for short English text such as tweets.

## Results on held-out test tweets

| Test tweets | Accuracy | Macro F1 | Precision (pos) | Recall (pos) |
| --- | --- | --- | --- | --- |
{row}

Confusion matrix (rows actual, columns predicted; negative, positive): {cm}

## Training

- 400,000 tweets from a stratified 500K subset of Sentiment140 (80/10/10 split, seed 42)
- 1 epoch, effective batch 32, learning rate 5e-5, max 48 tokens, on a laptop CPU (Ryzen 5 4500U)
- Labels in Sentiment140 come from emoticons, so some training labels are noisy

## Use

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{HUB_MODEL_ID}")
clf("Stuck in traffic for 3 hours")
```

Code, evaluation and REST API: https://github.com/ngugiauraa/sentiment-analyzer
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    if not metrics["model"].endswith("models/final"):
        sys.exit("reports/metrics.json is not for models/final. Run: python scripts/evaluate.py")

    api = HfApi()
    # Use readable labels in the pipeline output instead of LABEL_0 / LABEL_1.
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["id2label"] = {"0": "NEGATIVE", "1": "POSITIVE"}
    config["label2id"] = {"NEGATIVE": 0, "POSITIVE": 1}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(LOCAL_MODEL_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(metrics))

    api.create_repo(HUB_MODEL_ID, exist_ok=True)
    api.upload_folder(
        repo_id=HUB_MODEL_ID,
        folder_path=LOCAL_MODEL_DIR,
        ignore_patterns=["training_args.bin"],
        commit_message="Upload fine-tuned model",
    )
    print(f"Model: https://huggingface.co/{HUB_MODEL_ID}")

    if not args.model_only:
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=os.path.join(ROOT, "demo"),
                          commit_message="Deploy demo")
        print(f"Demo:  https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()

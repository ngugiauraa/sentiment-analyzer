"""
Evaluates a trained model on the held-out test split (data/processed/test.csv),
which the model never saw during training or checkpoint selection.

Run from the project root:
    python scripts/evaluate.py                      # models/final, full 50K test set
    python scripts/evaluate.py --model models/checkpoints/checkpoint-5000
    python scripts/evaluate.py --limit 2000         # quick check on a sample

Writes:
    reports/metrics.json          accuracy, precision, recall, F1, timing
    reports/confusion_matrix.png  for the README / portfolio
    reports/misclassified.csv     the most confident mistakes, for error analysis
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # run from anywhere

from sentiment.model import LABEL_NAMES, ROOT, SentimentModel, default_source  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "processed", "test.csv")
REPORTS_DIR = os.path.join(ROOT, "reports")


def save_confusion_matrix(cm, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=names)
    ax.set_yticks([0, 1], labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=default_source(), help="local folder or Hugging Face repo id")
    parser.add_argument("--limit", type=int, default=None, help="evaluate a stratified sample of this many rows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(TEST_PATH)
    df["text"] = df["text"].astype(str)
    if args.limit and args.limit < len(df):
        df = df.groupby("label").sample(n=args.limit // 2, random_state=42)
    print(f"Evaluating {args.model} on {len(df):,} test tweets...")

    model = SentimentModel(args.model, args.threads)
    started = time.time()
    preds, confs = model.predict(
        df["text"].tolist(), args.batch_size,
        on_batch=lambda done, total: print(f"\r  {done:,}/{total:,}", end="", flush=True),
    )
    seconds = time.time() - started
    print()

    y = df["label"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
    macro_f1 = precision_recall_fscore_support(y, preds, average="macro")[2]
    cm = confusion_matrix(y, preds, labels=[0, 1])
    metrics = {
        "model": os.path.relpath(args.model, ROOT).replace("\\", "/") if os.path.isdir(args.model) else args.model,
        "test_rows": int(len(df)),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision_positive": round(float(precision), 4),
        "recall_positive": round(float(recall), 4),
        "f1_positive": round(float(f1), 4),
        "f1_macro": round(float(macro_f1), 4),
        "confusion_matrix": {"labels": ["negative", "positive"], "rows_actual_cols_predicted": cm.tolist()},
        "inference_seconds": round(seconds, 1),
        "tweets_per_second": round(len(df) / seconds, 1),
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(cm, os.path.join(REPORTS_DIR, "confusion_matrix.png"))

    wrong = df.assign(predicted=preds, confidence=confs)[preds != y]
    wrong.sort_values("confidence", ascending=False).head(200).to_csv(os.path.join(REPORTS_DIR, "misclassified.csv"), index=False)

    print(classification_report(y, preds, target_names=["negative", "positive"], digits=4))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=["actual neg", "actual pos"], columns=["pred neg", "pred pos"]))
    print(f"\nAccuracy {metrics['accuracy']:.4f} | macro F1 {metrics['f1_macro']:.4f} | {metrics['tweets_per_second']} tweets/s")
    print("Saved reports/metrics.json, reports/confusion_matrix.png, reports/misclassified.csv")


if __name__ == "__main__":
    main()

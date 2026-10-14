"""
Labels new text with the trained model: CSV in, CSV out.

Run from the project root:
    python scripts/inference.py --input my_tweets.csv --output predictions.csv
    python scripts/inference.py --input reviews.csv --text-column review
    python scripts/inference.py --text "I love this" --text "worst day ever"

The output keeps every input column and adds:
    sentiment   "positive" or "negative"
    confidence  probability of that label, 0.5 to 1.0
"""

import argparse
import os
import sys

import pandas as pd

from predict_utils import DEFAULT_MODEL_DIR, LABEL_NAMES, load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="CSV file with a text column")
    parser.add_argument("--output", help="where to write results (default: <input>_predictions.csv)")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text", action="append", help="classify a single text (repeatable)")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.input and not args.text:
        parser.error("give --input FILE.csv or at least one --text")

    if args.text:
        df = pd.DataFrame({args.text_column: args.text})
    else:
        df = pd.read_csv(args.input)
        if args.text_column not in df.columns:
            sys.exit(f"Column '{args.text_column}' not found. Columns are: {', '.join(df.columns)}. Use --text-column.")

    texts = df[args.text_column].fillna("").astype(str).tolist()
    tokenizer, model = load_model(args.model, args.threads)
    labels, confidences = predict(texts, tokenizer, model, args.batch_size)

    df["sentiment"] = [LABEL_NAMES[int(label)] for label in labels]
    df["confidence"] = confidences.round(4)

    if args.text:
        for text, label, conf in zip(texts, df["sentiment"], df["confidence"]):
            print(f"{label:8s} {conf:.3f}  {text}")
        return

    output = args.output or os.path.splitext(args.input)[0] + "_predictions.csv"
    df.to_csv(output, index=False)
    counts = df["sentiment"].value_counts().to_dict()
    print(f"Labelled {len(df):,} rows -> {output}  ({counts})")


if __name__ == "__main__":
    main()

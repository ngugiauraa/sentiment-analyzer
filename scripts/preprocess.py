"""
Cleans and splits the Sentiment140 dataset for DistilBERT fine-tuning.

Run from the project root:
    python scripts/preprocess.py

Outputs:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sentiment140.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

COLUMN_NAMES = ["target", "id", "date", "flag", "user", "text"]
SUBSET_SIZE = 500_000

def clean_tweet(text: str) -> str:
    """
    Basic tweet text cleaning:
    - strip URLs (they carry no sentiment signal and add noise)
    - strip @mentions (usernames aren't sentiment-bearing tokens)
    - collapse repeated whitespace
    Deliberately KEEPS hashtags and punctuation/emoticons — those often
    carry real sentiment signal (e.g. ":)" or "#disappointed") that a
    model like DistilBERT can actually learn from, unlike a purely
    keyword-based classifier.
    """
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean():
    print("Loading raw CSV (1.6M rows — this will take a moment)...")
    df = pd.read_csv(
        RAW_PATH,
        encoding="latin-1",
        names=COLUMN_NAMES,
        header=None
    )
    print(f"  Loaded {len(df)} rows")

    df = df[df["target"].isin([0, 4])].copy()
    df["label"] = df["target"].map({0: 0, 4: 1})

    print("Cleaning tweet text...")
    df["clean_text"] = df["text"].apply(clean_tweet)

    df = df[df["clean_text"].str.len() > 0]

    df = df[["clean_text", "label"]].rename(columns={"clean_text": "text"})
    print(f"  {len(df)} rows after cleaning")
    print(f"  Label balance: {df['label'].value_counts().to_dict()}")

    # Subsample to SUBSET_SIZE, stratified to preserve the ~50/50 label
    # balance. DistilBERT fine-tuning saturates quickly on transfer
    # learning tasks — gains from 500K -> 1.28M examples are typically
    # small (often 1-2 accuracy points) while training time scales
    # roughly linearly, so this is a deliberate, documented time/accuracy
    # trade-off rather than an arbitrary cut.
    if SUBSET_SIZE and len(df) > SUBSET_SIZE:
        df, _ = train_test_split(
            df, train_size=SUBSET_SIZE, random_state=42, stratify=df["label"]
        )
        print(f"  Subsampled to {len(df)} rows (stratified)")
        print(f"  Subset label balance: {df['label'].value_counts().to_dict()}")

    return df


def split_and_save(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"\nSaved splits:")
    print(f"  train: {len(train_df)} rows -> data/processed/train.csv")
    print(f"  val:   {len(val_df)} rows -> data/processed/val.csv")
    print(f"  test:  {len(test_df)} rows -> data/processed/test.csv")


def main():
    df = load_and_clean()
    split_and_save(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
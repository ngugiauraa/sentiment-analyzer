"""
Fine-tunes DistilBERT for binary sentiment classification on the full
Sentiment140 dataset (~1.28M training tweets).

Designed for CPU-only, 16GB RAM hardware:
    - moderate batch size + gradient accumulation (efficient without
      risking memory pressure)
    - short max sequence length (tweets are short; saves memory + time)
    - checkpoints saved every N steps, with automatic resume

Run from the project root:
    python scripts/train.py

If interrupted (crash, restart, closed terminal), just run the same
command again — it will automatically resume from the latest checkpoint
in models/checkpoints/ instead of starting over.

Progress is logged both to the console AND to logs/train.log, so you can
close the terminal and check progress later with:
    tail -f logs/train.log
"""

import os
import logging
import glob

import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorWithPadding
import torch

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")
FINAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "final")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 48
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 1
SAVE_STEPS = 2000

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_datasets():
    logger.info("Loading processed CSVs...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    logger.info(f"  train: {len(train_ds)} examples, val: {len(val_ds)} examples")
    return train_ds, val_ds


def tokenize_datasets(train_ds, val_ds, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )

    logger.info("Tokenizing (batched, cached to disk by datasets library)...")
    train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=1000)
    val_ds = val_ds.map(tokenize_fn, batched=True, batch_size=1000)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds


def find_latest_checkpoint():
    """Check for an existing checkpoint to resume from. Returns the path
    to the latest one, or None if this is a fresh start."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.split("-")[-1]))
    return latest


def main():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    torch.set_num_threads(os.cpu_count())
    logger.info(f"Using {os.cpu_count()} CPU threads")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    train_ds, val_ds = load_datasets()
    train_ds, val_ds = tokenize_datasets(train_ds, val_ds, tokenizer)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_dir=LOG_DIR,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        no_cuda=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    resume_checkpoint = find_latest_checkpoint()
    if resume_checkpoint:
        logger.info(f"Found existing checkpoint — resuming from: {resume_checkpoint}")
    else:
        logger.info("No checkpoint found — starting fresh training run.")

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info("Training complete. Saving final model...")
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    logger.info(f"Final model saved to {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
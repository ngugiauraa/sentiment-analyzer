"""
Fine-tunes DistilBERT for binary sentiment classification on a 500K-tweet
stratified subset of Sentiment140 (400K train / 50K val / 50K test, made by
scripts/preprocess.py).

Designed for CPU-only, 16GB RAM hardware:
    - moderate batch size + gradient accumulation (efficient without
      risking memory pressure)
    - short max sequence length (48 tokens; tweets are short) with dynamic
      padding, so each batch is only as long as its longest tweet
    - checkpoints saved every 500 steps, with automatic resume
    - leaves CPU headroom (TRAIN_THREADS, default 4 of 6) so the laptop
      stays usable and runs cooler

Run from the project root:
    python scripts/train.py

If interrupted (crash, restart, closed terminal), just run the same
command again — it will automatically resume from the latest checkpoint
in models/checkpoints/ instead of starting over. Checkpoints are never
deleted by this script.

Safety checks before training starts:
    - if another training run is active (logs/train.lock holds a live PID),
      it exits instead of starting a second one
    - if models/final already exists, training is finished; it exits unless
      you pass --force

Progress is logged both to the console AND to logs/train.log, so you can
close the terminal and check progress later with:
    tail -f logs/train.log
"""

import os
import sys
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
SAVE_STEPS = 500
# Evaluating 50K tweets takes many minutes on CPU, so progress checks use a
# 5K slice of the validation set every 2,500 steps. The real score comes from
# scripts/evaluate.py on the untouched test set.
EVAL_STEPS = 2500
EVAL_SUBSET = 5000
NUM_THREADS = int(os.environ.get("TRAIN_THREADS", "4"))
LOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "train.lock")

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

    train_df["text"] = train_df["text"].astype(str)
    val_df["text"] = val_df["text"].astype(str)
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_df = val_df.sample(n=min(EVAL_SUBSET, len(val_df)), random_state=42)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

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


def pid_is_running(pid):
    """True if a process with this PID exists (works on Windows and Linux)."""
    try:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return code.value == 259  # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def guard_against_duplicate_runs():
    if os.path.isdir(FINAL_MODEL_DIR) and os.listdir(FINAL_MODEL_DIR) and "--force" not in sys.argv:
        logger.info(f"A finished model already exists in {FINAL_MODEL_DIR}. Not retraining (pass --force to train anyway).")
        sys.exit(0)
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH) as f:
            pid = int(f.read().strip() or 0)
        if pid and pid_is_running(pid):
            logger.info(f"Another training run is active (PID {pid}). Not starting a second one.")
            sys.exit(1)
        logger.info(f"Removing stale lock from PID {pid}, which is no longer running.")
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))


def main():
    guard_against_duplicate_runs()
    try:
        train()
    finally:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)


def train():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    torch.set_num_threads(NUM_THREADS)
    logger.info(f"Using {NUM_THREADS} of {os.cpu_count()} CPU threads (set TRAIN_THREADS to change)")

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
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_dir=LOG_DIR,
        logging_steps=100,
        report_to=[],
        no_cuda=True,
        # Worker processes on Windows each load a copy of torch and compete
        # for the same cores; the main process keeps up fine for short tweets.
        dataloader_num_workers=0,
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
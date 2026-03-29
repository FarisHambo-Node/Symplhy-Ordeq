"""
data_prep.py — Load the Emotion dataset from HuggingFace, clean & subsample.

Ordeq features demonstrated:
  • HuggingfaceDataset Input IO (load-only)
  • with_load_options() — pass split="train" to HF loader
  • @node chaining: HF → CSV → cleaned CSV
"""

import pandas as pd

from ordeq import node

from ordeq_showcase import catalog


# Emotion label mapping
EMOTION_LABELS = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "surprise", 5: "fear"}


# ─── Node 1: Download Emotion dataset from HuggingFace ───────────────────────
@node(
    inputs=catalog.emotion_hf,
    outputs=catalog.emotion_raw_csv,
)
def download_emotion(hf_dataset) -> pd.DataFrame:
    """Download the dair-ai/emotion dataset and convert to DataFrame."""
    # HuggingfaceDataset loads a DatasetDict; take the 'train' split
    if hasattr(hf_dataset, "keys"):
        # It's a DatasetDict — pick the train split
        split = hf_dataset["train"]
    else:
        split = hf_dataset

    df = split.to_pandas()

    # Map numeric labels to strings
    if "label" in df.columns:
        df["label_name"] = df["label"].map(EMOTION_LABELS)

    print(f"  📥 Downloaded Emotion dataset: {df.shape[0]} rows")
    print(f"     Labels: {df['label_name'].value_counts().to_dict()}")
    return df


# ─── Node 2: Clean & subsample ───────────────────────────────────────────────
@node(
    inputs=catalog.emotion_raw_csv,
    outputs=catalog.emotion_clean,
)
def clean_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text data and subsample for demo speed."""
    # Remove empty texts
    df = df[df["text"].str.strip().str.len() > 0].copy()

    # Subsample: take 200 per class for a fast demo (1200 total)
    df = (
        df.groupby("label_name", group_keys=False)
        .apply(lambda g: g.sample(n=min(200, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    # Basic text cleaning
    df["text_clean"] = (
        df["text"]
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)  # remove URLs
        .str.replace(r"@\w+", "", regex=True)      # remove mentions
        .str.replace(r"[^\w\s]", "", regex=True)    # remove punctuation
        .str.strip()
    )

    print(f"  🧹 Cleaned: {df.shape[0]} rows, {df['label_name'].nunique()} classes")
    return df

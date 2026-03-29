"""
nodes.py — LLM text classification pipeline node functions.

Ported from Ordeq's llm_pipeline/ package (data_prep.py, inference.py, analysis.py).

Key differences from Ordeq:
  • Ordeq: TransformersClassifier custom IO class loads models via .load()
  • Kedro: load_emotion_classifier() / load_embedding_model() are explicit nodes
    that return the loaded model as a MemoryDataset flowing to downstream nodes
  • Ordeq: SentenceTransformer IO from ordeq-sentence-transformers
  • Kedro: loaded inline in a node function
  • Parameters (model names, batch sizes) come from parameters.yml
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREP (was data_prep.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def download_emotion(emotion_hf, parameters: dict[str, Any]) -> pd.DataFrame:
    """Download the dair-ai/emotion dataset and convert to DataFrame."""
    emotion_labels = parameters["emotion_labels"]

    if hasattr(emotion_hf, "keys"):
        split = emotion_hf["train"]
    else:
        split = emotion_hf

    df = split.to_pandas()

    if "label" in df.columns:
        df["label_name"] = df["label"].map(emotion_labels)

    logger.info("📥 Downloaded Emotion dataset: %d rows", df.shape[0])
    logger.info("   Labels: %s", df["label_name"].value_counts().to_dict())
    return df


def clean_emotion(df: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    """Clean text data and subsample for demo speed."""
    emotion_labels = parameters["emotion_labels"]
    sample_per_class = parameters["sample_per_class"]

    df = df[df["text"].str.strip().str.len() > 0].copy()

    df["label_name"] = df["label"].map(emotion_labels)
    sampled_parts = []
    for label_name in df["label_name"].unique():
        group = df[df["label_name"] == label_name]
        sampled_parts.append(group.sample(n=min(sample_per_class, len(group)), random_state=42))
    df = pd.concat(sampled_parts).reset_index(drop=True)

    df["text_clean"] = (
        df["text"]
        .str.lower()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"@\w+", "", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )

    logger.info("🧹 Cleaned: %d rows, %d classes", df.shape[0], df["label_name"].nunique())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (was custom_io.py + inference.py catalog in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def load_emotion_classifier(parameters: dict[str, Any]):
    """Load HuggingFace text-classification pipeline.

    In Ordeq:  TransformersClassifier custom IO class with load() method
    In Kedro:  explicit node that loads and returns the model as MemoryDataset
    """
    from transformers import pipeline as hf_pipeline

    clf_params = parameters["classifier"]
    kwargs = {
        "task": clf_params["task"],
        "model": clf_params["model_name"],
        "device": clf_params["device"],
    }
    if clf_params.get("top_k") is not None:
        kwargs["top_k"] = clf_params["top_k"]

    logger.info("🤗 Loading model: %s", clf_params["model_name"])
    return hf_pipeline(**kwargs)


def load_embedding_model(parameters: dict[str, Any]):
    """Load SentenceTransformer model.

    In Ordeq:  SentenceTransformer IO from ordeq-sentence-transformers
    In Kedro:  explicit node function
    """
    from sentence_transformers import SentenceTransformer

    model_name = parameters["embedding_model"]["model"]
    logger.info("🧬 Loading SentenceTransformer: %s", model_name)
    return SentenceTransformer(model_name)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE (was inference.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_emotions(
    df: pd.DataFrame, classifier, parameters: dict[str, Any]
) -> list[dict]:
    """Run HuggingFace emotion classifier on each text."""
    texts = df["text_clean"].tolist()
    batch_size = parameters["batch_size"]
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = classifier(batch)
        for pred in preds:
            top = pred[0] if isinstance(pred, list) else pred
            results.append({
                "predicted_label": top["label"].lower(),
                "confidence": round(float(top["score"]), 4),
            })

    logger.info("🤖 Classified %d texts", len(results))
    return results


def generate_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    """Generate sentence embeddings using SentenceTransformer."""
    texts = df["text_clean"].tolist()
    emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    logger.info("🧬 Generated embeddings: shape %s", emb.shape)
    return emb


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS (was analysis.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_results(df: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """Merge model predictions with original data for analysis."""
    pred_df = pd.DataFrame(predictions)
    result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    correct = (result["label_name"] == result["predicted_label"]).sum()
    total = len(result)
    logger.info("📋 Results merged: %d/%d correct (%.1f%%)", correct, total, correct / total * 100)
    return result


def compute_llm_metrics(df: pd.DataFrame) -> dict:
    """Compute classification metrics for the LLM predictions."""
    y_true = df["label_name"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "per_class": {
            cls: {
                "precision": round(float(vals["precision"]), 4),
                "recall": round(float(vals["recall"]), 4),
                "f1_score": round(float(vals["f1-score"]), 4),
                "support": int(vals["support"]),
            }
            for cls, vals in report.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "model": "bhadresh-savani/distilbert-base-uncased-emotion",
    }

    logger.info("📊 LLM Accuracy: %.2f%%", metrics["accuracy"] * 100)
    logger.info("   Macro F1: %.4f", metrics["macro_avg_f1"])
    return metrics


def plot_embeddings(
    embeddings: np.ndarray, df: pd.DataFrame, parameters: dict[str, Any]
) -> plt.Figure:
    """Create t-SNE visualization of sentence embeddings colored by emotion."""
    tsne_params = parameters["tsne"]
    logger.info("🎨 Computing t-SNE (this may take a moment)...")
    tsne = TSNE(
        n_components=tsne_params["n_components"],
        random_state=tsne_params["random_state"],
        perplexity=tsne_params["perplexity"],
    )
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = df["label_name"].values
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=label, alpha=0.6, s=20,
        )

    ax.legend(title="Emotion", fontsize=9)
    ax.set_title("t-SNE of Sentence Embeddings (all-MiniLM-L6-v2)", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    logger.info("📈 t-SNE plot created")
    return fig

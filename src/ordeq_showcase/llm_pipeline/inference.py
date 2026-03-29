"""
inference.py — Run HuggingFace transformer model for text classification,
               and generate sentence embeddings.

Ordeq features demonstrated:
  • Custom IO class (TransformersClassifier) — build your own IO!
  • SentenceTransformer IO from ordeq-sentence-transformers
  • In-memory IO for intermediate results
"""

import pandas as pd
import numpy as np

from ordeq import node

from ordeq_showcase import catalog
from ordeq_showcase.custom_io import TransformersClassifier


# ── Catalog extensions: model IOs defined here for clarity ────────────────────
# Custom IO: HuggingFace text-classification pipeline
emotion_classifier = TransformersClassifier(
    model_name="j-hartmann/emotion-english-distilbert-roberta-base",
    task="text-classification",
    top_k=1,
).with_attributes(
    description="DistilRoBERTa emotion classifier (7 classes)",
    layer="model",
)

# SentenceTransformer IO from ordeq's built-in package
from ordeq_sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    model="all-MiniLM-L6-v2",
).with_attributes(
    description="Sentence-Transformer for generating text embeddings",
    layer="model",
)


# ─── Node 3: Run emotion classification ──────────────────────────────────────
@node(
    inputs=[catalog.emotion_clean, emotion_classifier],
    outputs=catalog.llm_predictions,
)
def classify_emotions(df: pd.DataFrame, classifier) -> list[dict]:
    """Run HuggingFace emotion classifier on each text."""
    texts = df["text_clean"].tolist()
    results = []

    # Process in batches for speed
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = classifier(batch)
        for pred in preds:
            # pred is a list of dicts like [{"label": "joy", "score": 0.98}]
            top = pred[0] if isinstance(pred, list) else pred
            results.append({
                "predicted_label": top["label"].lower(),
                "confidence": round(float(top["score"]), 4),
            })

    print(f"  🤖 Classified {len(results)} texts")
    return results


# ─── Node 4: Generate sentence embeddings ────────────────────────────────────
@node(
    inputs=[catalog.emotion_clean, embedding_model],
    outputs=catalog.embeddings,
)
def generate_embeddings(df: pd.DataFrame, model) -> np.ndarray:
    """Generate sentence embeddings using SentenceTransformer."""
    texts = df["text_clean"].tolist()
    emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"  🧬 Generated embeddings: shape {emb.shape}")
    return emb

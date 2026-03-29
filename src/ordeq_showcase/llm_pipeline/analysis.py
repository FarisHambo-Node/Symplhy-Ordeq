"""
analysis.py — Analyse LLM predictions, compute metrics, generate plots.

Ordeq features demonstrated:
  • YAML output for structured metrics
  • PandasCSV with with_save_options(index=False)
  • MatplotlibFigure for t-SNE embedding visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report

from ordeq import node

from ordeq_showcase import catalog


# ─── Node 5: Merge predictions with original data ────────────────────────────
@node(
    inputs=[catalog.emotion_clean, catalog.llm_predictions],
    outputs=catalog.llm_results_csv,
)
def merge_results(df: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    """Merge model predictions with original data for analysis."""
    pred_df = pd.DataFrame(predictions)
    result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    # Show a sample
    correct = (result["label_name"] == result["predicted_label"]).sum()
    total = len(result)
    print(f"  📋 Results merged: {correct}/{total} correct ({correct/total:.1%})")
    return result


# ─── Node 6: Compute LLM metrics ─────────────────────────────────────────────
@node(
    inputs=catalog.llm_results_csv,
    outputs=catalog.llm_metrics,
)
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
        "model": "j-hartmann/emotion-english-distilbert-roberta-base",
    }

    print(f"  📊 LLM Accuracy: {metrics['accuracy']:.2%}")
    print(f"     Macro F1: {metrics['macro_avg_f1']:.4f}")
    return metrics


# ─── Node 7: t-SNE embedding visualization ───────────────────────────────────
@node(
    inputs=[catalog.embeddings, catalog.emotion_clean],
    outputs=catalog.embedding_plot,
)
def plot_embeddings(embeddings: np.ndarray, df: pd.DataFrame) -> plt.Figure:
    """Create t-SNE visualization of sentence embeddings colored by emotion."""
    print("  🎨 Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = df["label_name"].values
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            label=label,
            alpha=0.6,
            s=20,
        )

    ax.legend(title="Emotion", fontsize=9)
    ax.set_title("t-SNE of Sentence Embeddings (all-MiniLM-L6-v2)", fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    print("  📈 t-SNE plot created")
    return fig

"""
evaluation.py — Evaluate the trained model, generate metrics and plots.

Ordeq features demonstrated:
  • Node producing structured YAML output (metrics)
  • MatplotlibFigure Output IO for saving plots
  • In-memory IO (IO[list]) for intermediate predictions
  • View node (read-only inspection, no output)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ordeq import node, Node, View

from ordeq_showcase import catalog


# ─── Node 6: Generate predictions ────────────────────────────────────────────
# Shows: in-memory IO (IO[list]) — predictions never touch disk
@node(
    inputs=[catalog.classifier, catalog.X_test],
    outputs=catalog.ml_predictions,
)
def predict(model: RandomForestClassifier, X_test: np.ndarray) -> list:
    """Generate predictions on the test set."""
    preds = model.predict(X_test).tolist()
    print(f"  🔮 Generated {len(preds)} predictions")
    return preds


# ─── Node 7: Compute metrics ─────────────────────────────────────────────────
# Shows: YAML IO for structured output
@node(
    inputs=[catalog.y_test, catalog.ml_predictions],
    outputs=catalog.ml_metrics,
)
def compute_metrics(y_test: np.ndarray, predictions: list) -> dict:
    """Compute classification metrics and save as YAML."""
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

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
    }

    print(f"  📊 Accuracy: {metrics['accuracy']:.2%}")
    return metrics


# ─── Node 8: Confusion matrix plot ───────────────────────────────────────────
# Shows: MatplotlibFigure Output IO
@node(
    inputs=[catalog.y_test, catalog.ml_predictions],
    outputs=catalog.confusion_matrix_plot,
)
def plot_confusion_matrix(y_test: np.ndarray, predictions: list) -> plt.Figure:
    """Create a confusion matrix heatmap and return the Figure."""
    labels = sorted(set(y_test.tolist()))
    cm = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Iris RandomForest")
    fig.tight_layout()

    print("  📈 Confusion matrix plot created")
    return fig


# ─── Node 9: Feature distribution plot ───────────────────────────────────────
@node(
    inputs=catalog.iris_features,
    outputs=catalog.feature_distribution_plot,
)
def plot_feature_distributions(df: pd.DataFrame) -> plt.Figure:
    """Plot histograms of all numeric features, colored by species."""
    numeric_cols = [c for c in df.columns if c != "species"]
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            for species in df["species"].unique():
                subset = df[df["species"] == species]
                axes[i].hist(subset[col], alpha=0.6, label=species, bins=15)
            axes[i].set_title(col, fontsize=10)
            axes[i].legend(fontsize=7)

    # Hide unused axes
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Species", fontsize=14)
    fig.tight_layout()
    print("  📊 Feature distribution plot created")
    return fig

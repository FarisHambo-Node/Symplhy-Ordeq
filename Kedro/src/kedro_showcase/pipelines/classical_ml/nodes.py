"""
nodes.py — Classical ML pipeline node functions.

Ported from Ordeq's classical_ml/ package (preprocessing.py, training.py, evaluation.py).

Key differences from Ordeq:
  • Ordeq: @node(inputs=catalog.X, outputs=catalog.Y) decorators wire IO
  • Kedro: plain functions — wiring happens in pipeline.py via node()
  • Ordeq: catalog.py Python objects for IO
  • Kedro: catalog.yml YAML definitions + parameters.yml for config
  • Parameters (test_size, n_estimators, etc.) are injected from parameters.yml
    instead of being hardcoded
"""

import re
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING (was preprocessing.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_col(name: str) -> str:
    """Convert CamelCase / mixed column names to snake_case, stripping units."""
    name = re.sub(r'(?i)cm$', '', name)
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    return name.strip().lower().replace(" ", "_")


def download_iris(iris_hf) -> pd.DataFrame:
    """Download Iris dataset from HuggingFace and convert to DataFrame.

    In Ordeq:  @node(inputs=catalog.iris_hf, outputs=catalog.iris_raw_csv)
    In Kedro:  inputs: "iris_hf" → outputs: "iris_raw_csv" (see pipeline.py)
    """
    if hasattr(iris_hf, "keys"):
        iris_hf = iris_hf["train"]
    df = iris_hf.to_pandas()
    logger.info("📥 Downloaded Iris: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_iris(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Iris dataset: drop nulls, normalize column names."""
    df.columns = [_normalize_col(c) for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c in ("unnamed:_0", "id")], errors="ignore")

    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        logger.info("🧹 Dropped %d null rows", before - after)

    if "species" in df.columns:
        label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        if df["species"].dtype in (int, np.int64, float):
            df["species"] = df["species"].map(label_map)

    logger.info("✅ Clean data: %d rows, columns: %s", df.shape[0], list(df.columns))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: ratios and interactions."""
    feature_cols = [c for c in df.columns if c != "species"]

    if "petal_length" in df.columns and "petal_width" in df.columns:
        df["petal_area"] = df["petal_length"] * df["petal_width"]
    if "sepal_length" in df.columns and "sepal_width" in df.columns:
        df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    if "petal_length" in df.columns and "sepal_length" in df.columns:
        df["petal_sepal_length_ratio"] = df["petal_length"] / df["sepal_length"]

    new_features = [c for c in df.columns if c not in feature_cols and c != "species"]
    logger.info("🔧 Engineered %d new features: %s", len(new_features), new_features)
    return df


def split_data(
    df: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test and scale features.

    In Ordeq:  hardcoded test_size=0.2, random_state=42
    In Kedro:  injected from params:classical_ml via parameters.yml
    """
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    feature_cols = [c for c in df.columns if c != "species"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["species"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info("✂️  Split: train=%d, test=%d", X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING (was training.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, parameters: dict[str, Any]
) -> RandomForestClassifier:
    """Train a RandomForest classifier on the Iris dataset.

    In Ordeq:  hardcoded model hyperparameters
    In Kedro:  injected from params:classical_ml.model via parameters.yml
    """
    model_params = parameters["model"]
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=parameters["random_state"],
        n_jobs=model_params["n_jobs"],
    )
    model.fit(X_train, y_train)

    logger.info(
        "🌲 Trained RandomForest: %d trees, max_depth=%s",
        model.n_estimators, model.max_depth
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION (was evaluation.py in Ordeq)
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model: RandomForestClassifier, X_test: np.ndarray) -> list:
    """Generate predictions on the test set.

    In Ordeq:  outputs=catalog.ml_predictions (IO[list], in-memory)
    In Kedro:  outputs="ml_predictions" → implicit MemoryDataset
    """
    preds = model.predict(X_test).tolist()
    logger.info("🔮 Generated %d predictions", len(preds))
    return preds


def compute_metrics(y_test: np.ndarray, predictions: list) -> dict:
    """Compute classification metrics and return as dict (saved as YAML)."""
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

    logger.info("📊 Accuracy: %.2f%%", metrics["accuracy"] * 100)
    return metrics


def plot_confusion_matrix(y_test: np.ndarray, predictions: list) -> plt.Figure:
    """Create a confusion matrix heatmap and return the Figure.

    In Ordeq:  outputs=catalog.confusion_matrix_plot (MatplotlibFigure IO)
    In Kedro:  outputs="confusion_matrix_plot" → matplotlib.MatplotlibWriter
    """
    labels = sorted(set(y_test.tolist()))
    cm = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Iris RandomForest")
    fig.tight_layout()

    logger.info("📈 Confusion matrix plot created")
    return fig


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

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Species", fontsize=14)
    fig.tight_layout()
    logger.info("📊 Feature distribution plot created")
    return fig

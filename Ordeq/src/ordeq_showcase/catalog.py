"""
catalog.py — Centralized IO definitions for the entire project.

This is the CATALOG: all data sources, models, and outputs defined in one place.
The key Ordeq principle: IO is separate from transformations.

Ordeq features demonstrated here:
  • IO, Input, Output classes
  • with_load_options()  — configure load behaviour
  • with_save_options()  — configure save behaviour
  • with_attributes()    — annotate IOs with metadata (layer, description)
  • Resources (@)        — mark IOs sharing the same underlying resource
"""

from pathlib import Path

from ordeq import IO, Input, Output
from ordeq_huggingface import HuggingfaceDataset
from ordeq_joblib import Joblib
from ordeq_showcase.matplotlib_figure import MatplotlibFigure
from ordeq_numpy import NumpyBinary
from ordeq_pandas import PandasCSV, PandasParquet
from ordeq_yaml import YAML

# ═══════════════════════════════════════════════════════════════════════════════
# ░░░ DATA LAYER PATHS ░░░
# ═══════════════════════════════════════════════════════════════════════════════

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
OUTPUTS = Path("outputs")

# ═══════════════════════════════════════════════════════════════════════════════
# ░░░ PIPELINE 1: CLASSICAL ML (Iris dataset + RandomForest) ░░░
# ═══════════════════════════════════════════════════════════════════════════════

# ── Raw data ──────────────────────────────────────────────────────────────────
# HuggingFace Input: only loading (Input, not IO)
iris_hf = HuggingfaceDataset(path="scikit-learn/iris").with_attributes(
    description="Iris flower dataset from HuggingFace Hub",
    layer="raw",
    source="huggingface",
)

# Raw CSV on disk — we save the HF dataset here, then reload from CSV
# Demonstrates: Resources (@) — same data, two different IOs
iris_raw_csv = PandasCSV(path=RAW / "iris.csv").with_attributes(
    description="Raw Iris dataset saved as CSV",
    layer="raw",
) @ "iris-data"

# ── Processed data ────────────────────────────────────────────────────────────
iris_clean = PandasCSV(path=PROCESSED / "iris_clean.csv").with_attributes(
    description="Cleaned Iris dataset (no nulls, correct types)",
    layer="processed",
)

iris_features = PandasParquet(path=PROCESSED / "iris_features.parquet").with_attributes(
    description="Engineered features for ML training",
    layer="features",
)

# ── Train / test splits (numpy arrays) ───────────────────────────────────────
X_train = NumpyBinary(path=PROCESSED / "X_train.npy").with_attributes(
    description="Training feature matrix", layer="model_input"
)
X_test = NumpyBinary(path=PROCESSED / "X_test.npy").with_attributes(
    description="Test feature matrix", layer="model_input"
)
y_train = NumpyBinary(path=PROCESSED / "y_train.npy").with_attributes(
    description="Training labels", layer="model_input"
)
y_test = NumpyBinary(path=PROCESSED / "y_test.npy").with_attributes(
    description="Test labels", layer="model_input"
)

# ── Model ─────────────────────────────────────────────────────────────────────
classifier = Joblib(path=MODELS / "random_forest.pkl").with_attributes(
    description="Trained RandomForest classifier",
    layer="model",
)

# ── Outputs ───────────────────────────────────────────────────────────────────
ml_metrics = YAML(path=OUTPUTS / "ml_metrics.yml").with_attributes(
    description="Classification metrics (accuracy, f1, etc.)",
    layer="reporting",
)

confusion_matrix_plot = MatplotlibFigure(path=OUTPUTS / "confusion_matrix.png").with_attributes(
    description="Confusion matrix heatmap",
    layer="reporting",
)

feature_distribution_plot = MatplotlibFigure(
    path=OUTPUTS / "feature_distributions.png"
).with_attributes(
    description="Feature distribution histograms",
    layer="reporting",
)

# ── Intermediate IO (in-memory, no file) ──────────────────────────────────────
# IO[type]() creates a transient, in-memory IO — data flows between nodes
# without ever hitting disk. This is great for intermediate results.
ml_predictions = IO[list]()


# ═══════════════════════════════════════════════════════════════════════════════
# ░░░ PIPELINE 2: LLM TEXT CLASSIFICATION (Emotion dataset + DistilBERT) ░░░
# ═══════════════════════════════════════════════════════════════════════════════

# ── Raw data ──────────────────────────────────────────────────────────────────
emotion_hf = HuggingfaceDataset(path="dair-ai/emotion").with_attributes(
    description="Emotion classification dataset (6 classes: joy, sadness, anger, fear, love, surprise)",
    layer="raw",
    source="huggingface",
)

emotion_raw_csv = PandasCSV(path=RAW / "emotion.csv").with_attributes(
    description="Emotion dataset saved as CSV",
    layer="raw",
)

# ── Processed data ────────────────────────────────────────────────────────────
emotion_clean = PandasCSV(path=PROCESSED / "emotion_clean.csv").with_attributes(
    description="Cleaned and subsampled emotion texts",
    layer="processed",
)

# ── In-memory intermediates ───────────────────────────────────────────────────
llm_predictions = IO[list]()
embeddings = IO[object]()

# ── Outputs ───────────────────────────────────────────────────────────────────
llm_metrics = YAML(path=OUTPUTS / "llm_metrics.yml").with_attributes(
    description="LLM classification accuracy & per-class metrics",
    layer="reporting",
)

llm_results_csv = PandasCSV(path=OUTPUTS / "llm_results.csv").with_save_options(
    index=False,
).with_attributes(
    description="Full predictions table: text, true label, predicted label",
    layer="reporting",
)

embedding_plot = MatplotlibFigure(path=OUTPUTS / "embedding_tsne.png").with_attributes(
    description="t-SNE visualization of sentence embeddings",
    layer="reporting",
)

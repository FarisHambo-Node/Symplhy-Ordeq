"""
preprocessing.py — Data loading, cleaning, feature engineering, train/test split.

Ordeq features demonstrated:
  • @node decorator with inputs / outputs
  • Multiple inputs and multiple outputs from a single node
  • IO decoupling: the node doesn't know WHERE data comes from or goes to
  • HuggingfaceDataset IO → PandasCSV IO chain
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ordeq import node

from ordeq_showcase import catalog


# ─── Node 1: Download from HuggingFace and save as CSV ───────────────────────
# Shows: HuggingfaceDataset (Input) → PandasCSV (IO)
# The node receives a HF Dataset object, converts to DataFrame, outputs it.
@node(
    inputs=catalog.iris_hf,
    outputs=catalog.iris_raw_csv,
)
def download_iris(hf_dataset) -> pd.DataFrame:
    """Download Iris dataset from HuggingFace and convert to DataFrame."""
    df = hf_dataset.to_pandas()
    print(f"  📥 Downloaded Iris: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─── Node 2: Clean the data ──────────────────────────────────────────────────
# Shows: simple transformation node, IO decoupling
@node(
    inputs=catalog.iris_raw_csv,
    outputs=catalog.iris_clean,
)
def clean_iris(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Iris dataset: drop nulls, normalize column names."""
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop any null rows
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"  🧹 Dropped {before - after} null rows")

    # Ensure species is a string label
    if "species" in df.columns:
        label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        if df["species"].dtype in (int, np.int64, float):
            df["species"] = df["species"].map(label_map)

    print(f"  ✅ Clean data: {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


# ─── Node 3: Feature engineering ─────────────────────────────────────────────
# Shows: intermediate IO (PandasParquet), transformation logic
@node(
    inputs=catalog.iris_clean,
    outputs=catalog.iris_features,
)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: ratios and interactions."""
    feature_cols = [c for c in df.columns if c != "species"]

    # Add petal area approximation and sepal area
    if "petal_length" in df.columns and "petal_width" in df.columns:
        df["petal_area"] = df["petal_length"] * df["petal_width"]
    if "sepal_length" in df.columns and "sepal_width" in df.columns:
        df["sepal_area"] = df["sepal_length"] * df["sepal_width"]

    # Add ratio features
    if "petal_length" in df.columns and "sepal_length" in df.columns:
        df["petal_sepal_length_ratio"] = df["petal_length"] / df["sepal_length"]

    new_features = [c for c in df.columns if c not in feature_cols and c != "species"]
    print(f"  🔧 Engineered {len(new_features)} new features: {new_features}")
    return df


# ─── Node 4: Train/test split ────────────────────────────────────────────────
# Shows: one node producing MULTIPLE outputs (4 numpy arrays)
@node(
    inputs=catalog.iris_features,
    outputs=[catalog.X_train, catalog.X_test, catalog.y_train, catalog.y_test],
)
def split_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test and scale features."""
    feature_cols = [c for c in df.columns if c != "species"]
    X = df[feature_cols].values
    y = df["species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  ✂️  Split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

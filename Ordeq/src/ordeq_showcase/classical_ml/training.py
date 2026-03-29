"""
training.py — Train a RandomForest classifier.

Ordeq features demonstrated:
  • Node with multiple inputs
  • Joblib IO for model serialization
  • Node receives numpy arrays, outputs a trained model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ordeq import node

from ordeq_showcase import catalog


# ─── Node 5: Train the model ─────────────────────────────────────────────────
# Shows: Joblib IO for saving sklearn models
@node(
    inputs=[catalog.X_train, catalog.y_train],
    outputs=catalog.classifier,
)
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForest classifier on the Iris dataset."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print(f"  🌲 Trained RandomForest: {model.n_estimators} trees, "
          f"max_depth={model.max_depth}")
    return model

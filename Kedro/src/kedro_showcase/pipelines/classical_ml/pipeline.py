"""
pipeline.py — Classical ML pipeline definition (Kedro wiring).

This is the Kedro equivalent of Ordeq's @node decorator wiring.

In Ordeq:
    @node(inputs=catalog.iris_hf, outputs=catalog.iris_raw_csv)
    def download_iris(hf_dataset): ...

In Kedro:
    node(func=download_iris, inputs="iris_hf", outputs="iris_raw_csv", name="download_iris")

The pipeline() function chains all nodes; Kedro resolves the DAG automatically
(just like Ordeq's run() resolves topological order).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    download_iris,
    clean_iris,
    engineer_features,
    split_data,
    train_random_forest,
    predict,
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_distributions,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # ── Preprocessing ─────────────────────────────────────────────────
        node(
            func=download_iris,
            inputs="iris_hf",
            outputs="iris_raw_csv",
            name="download_iris",
        ),
        node(
            func=clean_iris,
            inputs="iris_raw_csv",
            outputs="iris_clean",
            name="clean_iris",
        ),
        node(
            func=engineer_features,
            inputs="iris_clean",
            outputs="iris_features",
            name="engineer_features",
        ),
        # In Ordeq: outputs=[catalog.X_train, catalog.X_test, ...]
        # In Kedro: outputs=["X_train", "X_test", ...] — same idea!
        node(
            func=split_data,
            inputs=["iris_features", "params:classical_ml"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data",
        ),

        # ── Training ──────────────────────────────────────────────────────
        node(
            func=train_random_forest,
            inputs=["X_train", "y_train", "params:classical_ml"],
            outputs="classifier",
            name="train_random_forest",
        ),

        # ── Evaluation ────────────────────────────────────────────────────
        # In Ordeq: ml_predictions = IO[list]() — in-memory intermediate
        # In Kedro: "ml_predictions" not in catalog.yml → auto MemoryDataset
        node(
            func=predict,
            inputs=["classifier", "X_test"],
            outputs="ml_predictions",
            name="predict",
        ),
        node(
            func=compute_metrics,
            inputs=["y_test", "ml_predictions"],
            outputs="ml_metrics",
            name="compute_metrics",
        ),
        node(
            func=plot_confusion_matrix,
            inputs=["y_test", "ml_predictions"],
            outputs="confusion_matrix_plot",
            name="plot_confusion_matrix",
        ),
        node(
            func=plot_feature_distributions,
            inputs="iris_features",
            outputs="feature_distribution_plot",
            name="plot_feature_distributions",
        ),
    ])

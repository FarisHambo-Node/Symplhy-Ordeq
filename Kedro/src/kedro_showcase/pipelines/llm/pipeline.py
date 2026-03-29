"""
pipeline.py — LLM text classification pipeline definition (Kedro wiring).

Ported from Ordeq's llm_pipeline/ (data_prep.py, inference.py, analysis.py).

Key difference: In Ordeq, the TransformersClassifier and SentenceTransformer
were IO objects that auto-loaded via the catalog. In Kedro, we make them
explicit nodes (load_emotion_classifier, load_embedding_model) whose outputs
flow as MemoryDatasets to downstream nodes.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    download_emotion,
    clean_emotion,
    load_emotion_classifier,
    load_embedding_model,
    classify_emotions,
    generate_embeddings,
    merge_results,
    compute_llm_metrics,
    plot_embeddings,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # ── Data Prep ─────────────────────────────────────────────────────
        node(
            func=download_emotion,
            inputs=["emotion_hf", "params:llm"],
            outputs="emotion_raw_csv",
            name="download_emotion",
        ),
        node(
            func=clean_emotion,
            inputs=["emotion_raw_csv", "params:llm"],
            outputs="emotion_clean",
            name="clean_emotion",
        ),

        # ── Model Loading ─────────────────────────────────────────────────
        # In Ordeq: custom IO classes with load()
        # In Kedro: explicit nodes → MemoryDataset
        node(
            func=load_emotion_classifier,
            inputs="params:llm",
            outputs="emotion_classifier",
            name="load_emotion_classifier",
        ),
        node(
            func=load_embedding_model,
            inputs="params:llm",
            outputs="embedding_model",
            name="load_embedding_model",
        ),

        # ── Inference ─────────────────────────────────────────────────────
        node(
            func=classify_emotions,
            inputs=["emotion_clean", "emotion_classifier", "params:llm"],
            outputs="llm_predictions",
            name="classify_emotions",
        ),
        node(
            func=generate_embeddings,
            inputs=["emotion_clean", "embedding_model"],
            outputs="embeddings",
            name="generate_embeddings",
        ),

        # ── Analysis ──────────────────────────────────────────────────────
        node(
            func=merge_results,
            inputs=["emotion_clean", "llm_predictions"],
            outputs="llm_results_csv",
            name="merge_results",
        ),
        node(
            func=compute_llm_metrics,
            inputs="llm_results_csv",
            outputs="llm_metrics",
            name="compute_llm_metrics",
        ),
        node(
            func=plot_embeddings,
            inputs=["embeddings", "emotion_clean", "params:llm"],
            outputs="embedding_plot",
            name="plot_embeddings",
        ),
    ])

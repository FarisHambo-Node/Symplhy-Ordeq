"""
test_pipeline_structure.py — Verify pipeline DAGs are valid.

This is a Kedro-specific test pattern: verify that pipelines are properly
wired (no dangling inputs, correct node names, etc.)
"""

import pytest
from kedro.pipeline import Pipeline

from kedro_showcase.pipelines.classical_ml.pipeline import create_pipeline as create_classical_ml
from kedro_showcase.pipelines.llm.pipeline import create_pipeline as create_llm


class TestClassicalMlPipeline:
    def test_pipeline_creation(self):
        pipe = create_classical_ml()
        assert isinstance(pipe, Pipeline)

    def test_node_count(self):
        pipe = create_classical_ml()
        assert len(pipe.nodes) == 9  # 4 preprocessing + 1 training + 4 evaluation

    def test_node_names(self):
        pipe = create_classical_ml()
        names = {n.name for n in pipe.nodes}
        expected = {
            "download_iris", "clean_iris", "engineer_features", "split_data",
            "train_random_forest", "predict", "compute_metrics",
            "plot_confusion_matrix", "plot_feature_distributions",
        }
        assert names == expected

    def test_pipeline_inputs(self):
        pipe = create_classical_ml()
        # Only external inputs: iris_hf and params:classical_ml
        free_inputs = pipe.inputs()
        assert "iris_hf" in free_inputs
        assert "params:classical_ml" in free_inputs


class TestLlmPipeline:
    def test_pipeline_creation(self):
        pipe = create_llm()
        assert isinstance(pipe, Pipeline)

    def test_node_count(self):
        pipe = create_llm()
        assert len(pipe.nodes) == 9  # 2 data prep + 2 model load + 2 inference + 3 analysis

    def test_node_names(self):
        pipe = create_llm()
        names = {n.name for n in pipe.nodes}
        expected = {
            "download_emotion", "clean_emotion",
            "load_emotion_classifier", "load_embedding_model",
            "classify_emotions", "generate_embeddings",
            "merge_results", "compute_llm_metrics", "plot_embeddings",
        }
        assert names == expected


class TestDefaultPipeline:
    def test_combined_pipeline(self):
        ml = create_classical_ml()
        llm = create_llm()
        combined = ml + llm
        assert len(combined.nodes) == 18

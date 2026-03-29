"""
test_llm_pipeline.py — Unit tests for LLM pipeline nodes.

Ported from Ordeq's tests. Same pattern: nodes are plain functions,
call them directly with mock data.
"""

import pandas as pd
import pytest

from kedro_showcase.pipelines.llm.nodes import (
    clean_emotion,
    merge_results,
    compute_llm_metrics,
)


@pytest.fixture
def default_llm_parameters() -> dict:
    """Parameters matching parameters.yml llm section."""
    return {
        "sample_per_class": 200,
        "emotion_labels": {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "surprise", 5: "fear"},
        "classifier": {
            "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
            "task": "text-classification",
            "top_k": 1,
            "device": -1,
        },
        "embedding_model": {"model": "all-MiniLM-L6-v2"},
        "batch_size": 32,
        "tsne": {"n_components": 2, "perplexity": 30, "random_state": 42},
    }


@pytest.fixture
def sample_emotion_df() -> pd.DataFrame:
    return pd.DataFrame({
        "text": [
            "I am so happy today",
            "This makes me really sad",
            "I love you so much",
            "I am furious about this",
            "What a surprising turn of events",
            "I'm scared of the dark",
        ],
        "label": [1, 0, 2, 3, 4, 5],
        "label_name": ["joy", "sadness", "love", "anger", "surprise", "fear"],
    })


class TestCleanEmotion:
    def test_removes_empty_texts(self, default_llm_parameters):
        df = pd.DataFrame({
            "text": ["hello", "", "  ", "world"],
            "label": [1, 0, 2, 3],
            "label_name": ["joy", "sadness", "love", "anger"],
        })
        result = clean_emotion(df, default_llm_parameters)
        assert len(result) <= 2  # only "hello" and "world" have content

    def test_adds_text_clean_column(self, sample_emotion_df, default_llm_parameters):
        result = clean_emotion(sample_emotion_df, default_llm_parameters)
        assert "text_clean" in result.columns

    def test_text_is_lowered(self, sample_emotion_df, default_llm_parameters):
        result = clean_emotion(sample_emotion_df, default_llm_parameters)
        for text in result["text_clean"]:
            assert text == text.lower()


class TestMergeResults:
    def test_merge_adds_predicted_label(self, sample_emotion_df):
        predictions = [
            {"predicted_label": "joy", "confidence": 0.95},
            {"predicted_label": "sadness", "confidence": 0.88},
            {"predicted_label": "love", "confidence": 0.92},
            {"predicted_label": "anger", "confidence": 0.85},
            {"predicted_label": "surprise", "confidence": 0.80},
            {"predicted_label": "fear", "confidence": 0.78},
        ]
        result = merge_results(sample_emotion_df, predictions)
        assert "predicted_label" in result.columns
        assert "confidence" in result.columns
        assert len(result) == 6


class TestComputeLlmMetrics:
    def test_metrics_structure(self, sample_emotion_df):
        sample_emotion_df["predicted_label"] = sample_emotion_df["label_name"]
        metrics = compute_llm_metrics(sample_emotion_df)
        assert "accuracy" in metrics
        assert "per_class" in metrics
        assert "macro_avg_f1" in metrics

    def test_perfect_accuracy(self, sample_emotion_df):
        sample_emotion_df["predicted_label"] = sample_emotion_df["label_name"]
        metrics = compute_llm_metrics(sample_emotion_df)
        assert metrics["accuracy"] == 1.0

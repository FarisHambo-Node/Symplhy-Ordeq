"""
test_preprocessing.py — Unit tests for classical ML preprocessing nodes.

Ordeq features demonstrated:
  • Testing nodes: Ordeq nodes are just functions! Call them directly.
  • No need for a running pipeline — pass data in, assert on output.
  • This is the recommended pattern from Ordeq docs.
"""

import numpy as np
import pandas as pd
import pytest

from ordeq_showcase.classical_ml.preprocessing import (
    clean_iris,
    engineer_features,
    split_data,
)


@pytest.fixture
def sample_iris_df() -> pd.DataFrame:
    """Create a small synthetic Iris-like DataFrame for testing."""
    return pd.DataFrame({
        "SepalLength": [5.1, 4.9, 4.7, 7.0, 6.4, 6.9, 6.3, 5.8, 7.1, 6.3],
        "SepalWidth":  [3.5, 3.0, 3.2, 3.2, 3.2, 3.1, 3.3, 2.7, 3.0, 2.9],
        "PetalLength": [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 6.0, 5.1, 5.9, 5.6],
        "PetalWidth":  [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 2.5, 1.9, 2.1, 1.8],
        "Species":     [0,   0,   0,   1,   1,   1,   2,   2,   2,   2],
    })


class TestCleanIris:
    def test_columns_are_lowered(self, sample_iris_df):
        result = clean_iris(sample_iris_df)
        assert all(c == c.lower() for c in result.columns)

    def test_species_mapped_to_strings(self, sample_iris_df):
        result = clean_iris(sample_iris_df)
        assert result["species"].dtype == object
        assert set(result["species"].unique()) == {"setosa", "versicolor", "virginica"}

    def test_no_nulls(self, sample_iris_df):
        # Add a null row
        sample_iris_df.loc[10] = [None, None, None, None, None]
        result = clean_iris(sample_iris_df)
        assert result.isna().sum().sum() == 0


class TestEngineerFeatures:
    def test_adds_petal_area(self, sample_iris_df):
        cleaned = clean_iris(sample_iris_df)
        result = engineer_features(cleaned)
        assert "petal_area" in result.columns

    def test_adds_sepal_area(self, sample_iris_df):
        cleaned = clean_iris(sample_iris_df)
        result = engineer_features(cleaned)
        assert "sepal_area" in result.columns

    def test_adds_ratio(self, sample_iris_df):
        cleaned = clean_iris(sample_iris_df)
        result = engineer_features(cleaned)
        assert "petal_sepal_length_ratio" in result.columns


class TestSplitData:
    def test_returns_four_arrays(self, sample_iris_df):
        cleaned = clean_iris(sample_iris_df)
        featured = engineer_features(cleaned)
        result = split_data(featured)
        assert len(result) == 4
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)

    def test_split_proportions(self, sample_iris_df):
        cleaned = clean_iris(sample_iris_df)
        featured = engineer_features(cleaned)
        X_train, X_test, y_train, y_test = split_data(featured)
        total = len(X_train) + len(X_test)
        assert total == len(sample_iris_df)

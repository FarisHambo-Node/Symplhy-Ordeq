"""
huggingface_dataset.py — Custom Kedro AbstractDataset for HuggingFace datasets.

Ported from Ordeq's HuggingfaceDataset IO class (from ordeq-huggingface).

In Ordeq:
    iris_hf = HuggingfaceDataset(path="scikit-learn/iris")
    data = iris_hf.load()  # returns a HF Dataset

In Kedro:
    # catalog.yml
    iris_hf:
      type: kedro_showcase.datasets.huggingface_dataset.HuggingFaceDataset
      path: scikit-learn/iris

Kedro requires implementing:
  _load() → returns the data
  _save() → raises error (this is read-only, like Ordeq's Input)
  _describe() → returns metadata dict
"""

from typing import Any

from kedro.io import AbstractDataset


class HuggingFaceDataset(AbstractDataset):
    """Loads a dataset from HuggingFace Hub using the `datasets` library.

    This is a read-only dataset (like Ordeq's Input class).
    """

    def __init__(self, path: str, split: str | None = None, **kwargs):
        """
        Args:
            path: HuggingFace dataset path (e.g. "scikit-learn/iris")
            split: Optional split name (e.g. "train"). If None, returns DatasetDict.
        """
        self._path = path
        self._split = split
        self._kwargs = kwargs

    def _load(self) -> Any:
        from datasets import load_dataset
        return load_dataset(self._path, split=self._split)

    def _save(self, data: Any) -> None:
        raise NotImplementedError(
            "HuggingFaceDataset is read-only (like Ordeq's Input class). "
            "Use a different dataset type to save data."
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "path": self._path,
            "split": self._split,
        }

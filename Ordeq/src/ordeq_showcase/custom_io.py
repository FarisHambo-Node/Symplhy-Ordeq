"""
custom_io.py — Demonstrates how to create your own IO class.

Ordeq features demonstrated:
  • Custom IO by extending Input (load-only) or IO (load + save)
  • The IO just needs load() and/or save() methods
  • @dataclass(frozen=True) pattern for immutable IO definitions
  • with_attributes() works on custom IOs too
"""

from dataclasses import dataclass, field
from typing import Any

from ordeq import Input


@dataclass(frozen=True, kw_only=True)
class TransformersClassifier(Input):
    """Custom IO that loads a HuggingFace text-classification pipeline.

    This is an Input (load-only) because we don't save the model — we just
    load it for inference. Under the hood it uses transformers.pipeline().

    Usage:
        classifier = TransformersClassifier(
            model_name="j-hartmann/emotion-english-distilbert-roberta-base",
            task="text-classification",
            top_k=1,
        )
        # The model is NOT loaded yet — just defined
        model = classifier.load()   # now it loads
        result = model("I am so happy!")
    """

    model_name: str
    task: str = "text-classification"
    top_k: int | None = None
    device: str | int = -1  # -1 = CPU, 0 = first GPU

    def load(self, **load_options) -> Any:
        """Load a HuggingFace transformers pipeline."""
        from transformers import pipeline as hf_pipeline

        kwargs = {
            "task": self.task,
            "model": self.model_name,
            "device": self.device,
        }
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k

        kwargs.update(load_options)
        print(f"  🤗 Loading model: {self.model_name}")
        return hf_pipeline(**kwargs)

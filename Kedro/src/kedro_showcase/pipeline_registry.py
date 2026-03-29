"""
pipeline_registry.py — Register all pipelines for Kedro discovery.

In Ordeq, pipelines were auto-discovered via @node decorators in modules.
In Kedro, we explicitly register pipelines here. The '__default__' key
runs all pipelines together (like Ordeq's run(module)).
"""

from kedro.pipeline import Pipeline

from kedro_showcase.pipelines.classical_ml.pipeline import create_pipeline as create_classical_ml
from kedro_showcase.pipelines.llm.pipeline import create_pipeline as create_llm


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        A mapping from pipeline name to Pipeline object.
        '__default__' runs all pipelines when you do `kedro run`.
    """
    classical_ml = create_classical_ml()
    llm = create_llm()

    return {
        "classical_ml": classical_ml,
        "llm": llm,
        "__default__": classical_ml + llm,
    }

"""
viz.py — Standalone script to generate pipeline visualizations.

Usage:
    python viz.py

Ordeq features demonstrated:
  • viz() with multiple output formats (mermaid, mermaid-md)
  • Visualizing entire packages at once
"""

from pathlib import Path

from ordeq_viz import viz


def main():
    # Import all pipeline modules so @node decorators are registered
    from ordeq_showcase.classical_ml import preprocessing, training, evaluation  # noqa: F401
    from ordeq_showcase.llm_pipeline import data_prep, inference, analysis  # noqa: F401
    from ordeq_showcase import classical_ml, llm_pipeline

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # ── Classical ML Pipeline ─────────────────────────────────────────────
    viz(
        classical_ml,
        fmt="mermaid",
        output=output_dir / "classical_ml_pipeline.mermaid",
    )
    print("✅ Classical ML diagram saved")

    # ── LLM Pipeline ─────────────────────────────────────────────────────
    viz(
        llm_pipeline,
        fmt="mermaid",
        output=output_dir / "llm_pipeline.mermaid",
    )
    print("✅ LLM pipeline diagram saved")

    # ── Print to console ─────────────────────────────────────────────────
    print("\n── Classical ML Pipeline (Mermaid) ──")
    print(viz(classical_ml, fmt="mermaid"))

    print("\n── LLM Pipeline (Mermaid) ──")
    print(viz(llm_pipeline, fmt="mermaid"))


if __name__ == "__main__":
    main()

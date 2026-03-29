"""
__main__.py — Entry point for running the Ordeq showcase pipelines.

Run with:
    python -m ordeq_showcase              # run both pipelines
    python -m ordeq_showcase --ml         # run only classical ML pipeline
    python -m ordeq_showcase --llm        # run only LLM pipeline
    python -m ordeq_showcase --viz-only   # generate diagrams without running

Ordeq features demonstrated:
  • run() — execute nodes in topological order
  • viz() — generate Mermaid pipeline diagrams
  • hooks= parameter for injecting TimingHook and PipelineLogHook
  • Running modules / packages as runnables
"""

import argparse
import logging
import sys
from pathlib import Path

from ordeq import run
from ordeq_viz import viz

from ordeq_showcase.hooks import PipelineLogHook, TimingHook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)


def run_classical_ml() -> None:
    """Run the classical ML pipeline (Iris + RandomForest)."""
    # Import the pipeline package — Ordeq auto-discovers all @node decorators
    from ordeq_showcase.classical_ml import preprocessing, training, evaluation  # noqa: F401
    from ordeq_showcase import classical_ml

    timer = TimingHook()
    log_hook = PipelineLogHook("Classical ML Pipeline")

    run(classical_ml, hooks=[timer, log_hook])
    print(timer.summary())


def run_llm_pipeline() -> None:
    """Run the LLM text classification pipeline (Emotion + DistilRoBERTa)."""
    from ordeq_showcase.llm_pipeline import data_prep, inference, analysis  # noqa: F401
    from ordeq_showcase import llm_pipeline

    timer = TimingHook()
    log_hook = PipelineLogHook("LLM Text Classification Pipeline")

    run(llm_pipeline, hooks=[timer, log_hook])
    print(timer.summary())


def generate_diagrams() -> None:
    """Generate Mermaid pipeline diagrams for both pipelines."""
    from ordeq_showcase.classical_ml import preprocessing, training, evaluation  # noqa: F401
    from ordeq_showcase.llm_pipeline import data_prep, inference, analysis  # noqa: F401
    from ordeq_showcase import classical_ml, llm_pipeline

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Classical ML diagram
    ml_diagram = viz(classical_ml, fmt="mermaid", output=output_dir / "classical_ml_pipeline.mermaid")
    print(f"  📊 Classical ML diagram → {output_dir / 'classical_ml_pipeline.mermaid'}")

    # LLM diagram
    llm_diagram = viz(llm_pipeline, fmt="mermaid", output=output_dir / "llm_pipeline.mermaid")
    print(f"  📊 LLM diagram → {output_dir / 'llm_pipeline.mermaid'}")

    # Also print to console for quick preview
    print("\n── Classical ML Pipeline ──")
    ml_str = viz(classical_ml, fmt="mermaid")
    print(ml_str)

    print("\n── LLM Pipeline ──")
    llm_str = viz(llm_pipeline, fmt="mermaid")
    print(llm_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordeq Showcase — ML & LLM Pipelines")
    parser.add_argument("--ml", action="store_true", help="Run only the classical ML pipeline")
    parser.add_argument("--llm", action="store_true", help="Run only the LLM pipeline")
    parser.add_argument("--viz-only", action="store_true", help="Generate diagrams only (no execution)")
    args = parser.parse_args()

    if args.viz_only:
        generate_diagrams()
        return

    run_ml = args.ml or (not args.ml and not args.llm)
    run_llm = args.llm or (not args.ml and not args.llm)

    if run_ml:
        run_classical_ml()

    if run_llm:
        run_llm_pipeline()

    # Always generate diagrams after running
    generate_diagrams()

    print("\n🎉 All done! Check the outputs/ folder for results.")


if __name__ == "__main__":
    main()

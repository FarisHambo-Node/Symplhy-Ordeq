"""Kedro Showcase project entry point.

Run with:
    kedro run                         # run all pipelines
    kedro run --pipeline classical_ml # run only classical ML pipeline
    kedro run --pipeline llm          # run only LLM pipeline
    kedro viz run                     # launch Kedro-Viz dashboard
"""

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project

from pathlib import Path
import sys


def main():
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    from kedro.framework.cli import main as kedro_main
    kedro_main()


if __name__ == "__main__":
    main()

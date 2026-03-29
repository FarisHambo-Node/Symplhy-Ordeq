"""
hooks.py — Kedro lifecycle hooks.

Ported from Ordeq's hooks.py (TimingHook, PipelineLogHook).

Key differences:
  • Ordeq: NodeHook / RunHook protocols with before_node_run, after_node_run, etc.
  • Kedro: @hook_impl decorator on methods matching Kedro's hook specs
  • Ordeq hooks receive node objects; Kedro hooks receive node, catalog, inputs, etc.
  • Registered in settings.py HOOKS tuple (Ordeq: passed to run(hooks=[...]))
"""

import time
import logging
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)


class TimingHook:
    """Measures and logs execution time for each node.

    Ordeq equivalent:
        before_node_run(node) / after_node_run(node) / on_node_call_error(node, error)
    Kedro equivalent:
        before_node_run(...) / after_node_run(...) / on_node_error(...)
    """

    def __init__(self):
        self._start_times: dict[str, float] = {}
        self._timings: list[tuple[str, float]] = []

    @hook_impl
    def before_node_run(self, node: Node) -> None:
        self._start_times[node.name] = time.time()

    @hook_impl
    def after_node_run(self, node: Node, outputs: dict[str, Any]) -> None:
        elapsed = time.time() - self._start_times.pop(node.name, time.time())
        self._timings.append((node.name, elapsed))
        logger.info("⏱  %s: %.2fs", node.name, elapsed)

    @hook_impl
    def on_node_error(self, node: Node, error: Exception) -> None:
        elapsed = time.time() - self._start_times.pop(node.name, time.time())
        logger.error("❌ %s FAILED after %.2fs: %s", node.name, elapsed, error)

    def summary(self) -> str:
        """Return a summary of all node timings."""
        if not self._timings:
            return "No timings recorded."
        lines = ["\n⏱  Pipeline Timing Summary:"]
        lines.append("  " + "─" * 50)
        total = 0.0
        for name, elapsed in self._timings:
            lines.append(f"  {name:<35s} {elapsed:>7.2f}s")
            total += elapsed
        lines.append("  " + "─" * 50)
        lines.append(f"  {'TOTAL':<35s} {total:>7.2f}s")
        return "\n".join(lines)


class PipelineLogHook:
    """Logs pipeline start/end events.

    Ordeq equivalent:
        before_run(graph) / after_run(graph) — RunHook protocol
    Kedro equivalent:
        before_pipeline_run(...) / after_pipeline_run(...)
    """

    def __init__(self, pipeline_name: str = "Pipeline"):
        self.pipeline_name = pipeline_name

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any]) -> None:
        pipeline_name = run_params.get("pipeline_name") or self.pipeline_name
        logger.info("═" * 60)
        logger.info("🚀 Starting %s", pipeline_name)
        logger.info("═" * 60)

    @hook_impl
    def after_pipeline_run(self, run_params: dict[str, Any]) -> None:
        pipeline_name = run_params.get("pipeline_name") or self.pipeline_name
        logger.info("═" * 60)
        logger.info("✅ %s completed successfully!", pipeline_name)
        logger.info("═" * 60)

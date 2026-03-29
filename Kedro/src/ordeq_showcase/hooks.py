"""
hooks.py — Custom hooks for pipeline execution.

Ordeq features demonstrated:
  • NodeHook protocol: inject logic before/after each node runs
  • RunHook protocol: inject logic before/after the entire pipeline run
  • Hooks are passed to run() via the hooks= parameter
"""

import time


class TimingHook:
    """Measures and prints execution time for each node.

    Implements the NodeHook protocol:
      - before_node_run(node)
      - after_node_run(node)
      - on_node_call_error(node, error)
    """

    def __init__(self):
        self._start_times: dict[str, float] = {}
        self._timings: list[tuple[str, float]] = []

    def before_node_run(self, node) -> None:
        name = node.func_name
        self._start_times[name] = time.time()

    def after_node_run(self, node) -> None:
        name = node.func_name
        elapsed = time.time() - self._start_times.pop(name, time.time())
        self._timings.append((name, elapsed))
        print(f"  ⏱  {name}: {elapsed:.2f}s")

    def on_node_call_error(self, node, error) -> None:
        name = node.func_name
        elapsed = time.time() - self._start_times.pop(name, time.time())
        print(f"  ❌ {name} FAILED after {elapsed:.2f}s: {error}")

    def summary(self) -> str:
        """Print a summary of all node timings."""
        if not self._timings:
            return "No timings recorded."
        lines = ["\n⏱  Pipeline Timing Summary:"]
        lines.append("  " + "─" * 50)
        total = 0.0
        for name, elapsed in self._timings:
            # Extract just the function name (after the last ':')
            short_name = name.split(":")[-1] if ":" in name else name
            lines.append(f"  {short_name:<35s} {elapsed:>7.2f}s")
            total += elapsed
        lines.append("  " + "─" * 50)
        lines.append(f"  {'TOTAL':<35s} {total:>7.2f}s")
        return "\n".join(lines)


class PipelineLogHook:
    """Logs pipeline start/end events.

    Implements the RunHook protocol:
      - before_run(graph)
      - after_run(graph)
    """

    def __init__(self, pipeline_name: str = "Pipeline"):
        self.pipeline_name = pipeline_name

    def before_run(self, graph) -> None:
        n_nodes = len(list(graph.edges))
        print(f"\n{'═' * 60}")
        print(f"  🚀 Starting {self.pipeline_name} ({n_nodes} nodes)")
        print(f"{'═' * 60}")

    def after_run(self, graph) -> None:
        print(f"\n{'═' * 60}")
        print(f"  ✅ {self.pipeline_name} completed successfully!")
        print(f"{'═' * 60}")

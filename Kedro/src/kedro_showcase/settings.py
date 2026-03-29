"""
settings.py — Kedro project settings.

Equivalent to Ordeq's hooks= parameter in run() — hooks are registered here
so Kedro automatically applies them to every pipeline run.
"""

from kedro_showcase.hooks import TimingHook, PipelineLogHook

# Register hooks — these run before/after every node and pipeline execution.
# In Ordeq these were passed to run(hooks=[...]); Kedro uses this tuple.
HOOKS = (TimingHook(), PipelineLogHook())

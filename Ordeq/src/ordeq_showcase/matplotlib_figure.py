"""Matplotlib Figure IO for Ordeq 1.4+.

PyPI ``ordeq-matplotlib`` 1.0.0 imports removed ``ordeq.framework``; this uses the public API.
Do not add ``from __future__ import annotations`` — Ordeq inspects ``save`` signatures.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from ordeq import Output
from ordeq.types import PathLike


@dataclass(frozen=True, kw_only=True)
class MatplotlibFigure(Output[plt.Figure]):
    """IO to save matplotlib Figures."""

    path: PathLike

    def save(self, fig: plt.Figure) -> None:
        with self.path.open(mode="wb") as fh:
            fig.savefig(fh, format=os.path.splitext(str(self.path))[1][1:])  # noqa: PTH122

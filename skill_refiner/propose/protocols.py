"""Data types shared by the proposal and merge stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Polarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class ClusterProposal:
    """Analysis produced by one per-cluster LLM call."""

    cluster_id: int
    theme: str
    reinforce: list[str]
    soften: list[str]
    suggested_edit: str
    n_traces: int      # traces shown to the per-cluster LLM (context-limited)
    cluster_size: int  # true number of items in the cluster before budget truncation
    polarity: Polarity | None = None  # partition polarity (positive / negative)

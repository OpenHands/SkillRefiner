from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openhands.sdk.skills import Skill  # noqa: TC002

from skill_refiner.cluster import SingleClusterer
from skill_refiner.cluster.umap_hdbscan import (
    EmbeddingClient,
    cluster_by_umap_hdbscan,
    embed_items,
    group_items_by_cluster,
)
from skill_refiner.propose.core import RefinementContext  # noqa: TC001
from skill_refiner.propose.protocols import ClusterProposal

if TYPE_CHECKING:
    from skill_refiner.summarize.core import TraceSummary

__all__ = ["ClusterProposal", "ClusteredTraceDrivenRefiner"]

# Resolved relative to this file (repo root / .cache / ...) rather than the
# process's current working directory, so the default doesn't silently write a
# .cache/ directory wherever the caller happened to invoke the script from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "skill_refiner" / "summary_clustering"

_DEFAULT_MAX_CONCURRENT_PROPOSALS = 16
_DEFAULT_EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
_DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")


# ---------------------------------------------------------------------------
# Shared helpers (used by the per-cluster proposal prompts and the merge)
# ---------------------------------------------------------------------------

def _summary_block(s: TraceSummary) -> str:
    obs = "; ".join(s.observations) if s.observations else "(none)"
    return f"- trace {s.trace_id}: {s.summary}\n  observations: {obs}"


def _extract_json(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner)
    # Strip a bare language label (e.g. "json\n{...}") returned by some models
    if re.match(r"^[a-z0-9+\-]+\n", stripped):
        stripped = stripped.split("\n", 1)[1]
    # Recover when the model omits the opening `{"` of a JSON object (it treats
    # the prompt's JSON template as a prefix and continues from inside, e.g.
    # 'theme":"...","reinforce":[...]}')
    if not stripped.startswith(("{", "[", '"')):
        # e.g. 'theme":"value",...}' → '{"theme":"value",...}'
        stripped = '{"' + stripped
    elif stripped.startswith('"') and not stripped.startswith('"{'):
        # e.g. '"theme":"value",...}' → '{"theme":"value",...}'
        stripped = "{" + stripped
    return stripped


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class SummaryTextExtractor:
    """TextExtractor for trace summaries: the text that gets embedded."""

    def trace_id(self, item: TraceSummary) -> str:
        return item.trace_id

    def embed_text(self, item: TraceSummary) -> str:
        obs = "; ".join(item.observations) if item.observations else ""
        parts = [item.summary]
        if obs:
            parts.append(obs)
        return "\n".join(parts).strip()


class UmapHdbscanClusterer:
    """Clusterer backed by UMAP+HDBSCAN (default for method='umap_hdbscan')."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def cluster(self, vectors: dict[str, list[float]]) -> dict[str, int]:
        return cluster_by_umap_hdbscan(vectors, **self._kwargs)


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class ClusteredTraceDrivenRefiner:
    """Embeds one partition's summaries, clusters them, and proposes one edit per cluster.

    Paper §2.3–2.5: ``cluster_refiner`` (``PolarityClusterRefiner``) owns the
    polarity-specific proposal prompt and, on the negative partition, the evidence
    gate. The proposals are returned to the caller, which merges both partitions
    into the skill (``merge_polarity_proposals``).
    """

    def __init__(
        self,
        *,
        cluster_refiner: Any,
        embedding_base_url: str = _DEFAULT_EMBEDDING_BASE_URL,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", ""),
        method: str = "umap_hdbscan",
        max_embed_workers: int = 4,
        cache_dir: Path | None = None,
        clusterer: Any = None,
        max_concurrent_proposals: int = _DEFAULT_MAX_CONCURRENT_PROPOSALS,
    ) -> None:
        self._embedding_base_url = embedding_base_url
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key
        self._max_embed_workers = max_embed_workers
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._extractor = SummaryTextExtractor()
        if clusterer is not None:
            self._clusterer = clusterer
        elif method == "single":
            self._clusterer = SingleClusterer()
        elif method == "umap_hdbscan":
            self._clusterer = UmapHdbscanClusterer()
        else:
            raise ValueError(
                f"unknown clustering method {method!r}; "
                "this artifact ships only 'single' and 'umap_hdbscan'"
            )
        self._cluster_refiner = cluster_refiner
        self._max_concurrent_proposals = max(1, int(max_concurrent_proposals))

    def _build_embedding_client(self) -> EmbeddingClient:
        return EmbeddingClient(
            base_url=self._embedding_base_url,
            model=self._embedding_model,
            api_key=self._embedding_api_key,
        )

    async def propose_from_summaries(
        self,
        skill: Skill,
        summaries: list[TraceSummary],
        ctx: RefinementContext,
    ) -> list[ClusterProposal]:
        """Embed → cluster → one proposal per cluster. Returns the surviving proposals."""
        if len(summaries) < 3:
            return []

        vectors = embed_items(
            summaries,
            self._extractor,
            client=self._build_embedding_client(),
            cache_dir=self._cache_dir,
            max_workers=self._max_embed_workers,
        )
        if not vectors:
            return []

        assignments = self._clusterer.cluster(vectors)
        groups = group_items_by_cluster(summaries, assignments, self._extractor)
        if not groups:
            groups = {-1: list(summaries)}

        model = getattr(ctx.llm, "model", "") or ""

        sem = asyncio.Semaphore(self._max_concurrent_proposals)

        async def _guarded(cluster_id: int, members: list[Any]) -> ClusterProposal | None:
            async with sem:
                return await self._cluster_refiner.propose(cluster_id, members, skill, ctx, model)

        results = await asyncio.gather(*[
            _guarded(cluster_id, members) for cluster_id, members in groups.items()
        ])
        return [p for p in results if p is not None]

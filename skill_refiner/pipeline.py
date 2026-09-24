"""End-to-end SkillRefiner loop: summarize -> cluster -> propose -> merge.

Implements the SkillRefiner method (paper §2): feedback-augmented summarization,
outcome-partitioned UMAP+HDBSCAN clustering, per-cluster proposals, an evidence gate
on failure-derived proposals, and a cross-polarity merge (prompt: paper §A.5.4).
Defaults are the configuration of record. Each ablation overrides exactly one field.

Ported from ``scripts/experiments/run_summarizer_ablation.py`` (per-partition
summarize/cluster/propose) and ``scripts/refine/merge_signal_passes.py`` (the
cross-polarity merge that turns two partitions' cluster proposals into one
final skill) in the research repo. The two were separate CLI scripts there;
``RefineConfig`` unifies them into a single entry point, ``refine()``, that
returns the merged proposal directly instead of writing it to a sibling
directory tree.
"""

import datetime
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from openhands.sdk.llm import LLM
from openhands.sdk.skills import Skill
from pydantic import SecretStr

from skill_refiner.merge import merge_polarity_proposals
from skill_refiner.propose import (
    ClusteredTraceDrivenRefiner,
    PolarityClusterRefiner,
    RefinementContext,
    RefinementProposal,
    UmapHdbscanClusterer,
)
from skill_refiner.propose.protocols import ClusterProposal
from skill_refiner.summarize.core import TraceSummarizer, TraceSummary
from skill_refiner.summarize.llm import LLMTraceSummarizer
from skill_refiner.summarize.no_summary import NoSummaryTraceSummarizer
from skill_refiner.summarize.runner import TraceSummarizationRunner
from skill_refiner.trace.context import EvaluationContext
from skill_refiner.trace.serialization import serialize_trace
from skill_refiner.trace.store import Trace, TraceEvent

Summarizer = Literal["full_trace", "none"]
ClusterMethod = Literal["umap_hdbscan", "single"]


@dataclass
class RefineConfig:
    """Inputs and stage selection for one refine run."""

    skill_file: Path
    skill_name: str
    raw_jsonl: Path
    binary_rewards: Path
    output_dir: Path
    ground_truth_file: Path | None = None

    # Stage selection — the configuration of record.
    summarizer: Summarizer = "full_trace"
    cluster_method: ClusterMethod = "umap_hdbscan"
    min_cluster_size: int = 2
    verify_negative_clusters: bool = True
    use_positive: bool = True
    use_negative: bool = True

    # Models.
    model: str = "openai/gpt-5.4-mini"
    llm_provider: str = "eval_proxy"
    embedding_model: str = "qwen3-embedding:4b"
    embedding_base_url: str = "http://localhost:11434/v1"
    limit: int | None = None


# ---------------------------------------------------------------------------
# LLM config
#
# The source script resolved this via --secrets-file / --api-key / --base-url
# argparse flags and a litellm.register_model() call for a custom MiniMax
# model spec (``_resolve_llm_config``, ``_LLMConfig``, ``_register_extra_models``
# at run_summarizer_ablation.py:126-224). None of that is a RefineConfig field,
# so it is not ported; this is new, deliberately smaller glue code instead.
# ---------------------------------------------------------------------------

# No deployment URL is baked in here: the original values were this artifact's
# author's org-internal proxy endpoints, unreachable (and identifying) outside
# that org. Each provider's base URL is resolved at call time from an env var,
# then artifact.local.json (see artifact.local.example.json), with no fallback
# to a guessed default -- _build_llm raises loudly, naming the provider and how
# to configure it, rather than silently trying a placeholder host.
# Only the provider the configuration of record actually used is shipped. An
# unknown llm_provider still fails loudly in _build_llm, naming what is valid.
_PROVIDER_BASE_URL_ENV_VARS = {
    "eval_proxy": "SKILL_REFINER_BASE_URL_EVAL_PROXY",
}
_ARTIFACT_LOCAL_CONFIG_FILE = Path(__file__).resolve().parents[1] / "artifact.local.json"
_API_KEY_ENV_CANDIDATES = ("LLM_API_KEY", "OPENAI_API_KEY")


def _load_local_config() -> dict[str, Any]:
    if not _ARTIFACT_LOCAL_CONFIG_FILE.is_file():
        return {}
    try:
        data = json.loads(_ARTIFACT_LOCAL_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_provider_base_url(provider: str) -> str:
    env_var = _PROVIDER_BASE_URL_ENV_VARS[provider]
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    local = _load_local_config().get("llm_base_urls", {})
    val = str(local.get(provider, "")).strip() if isinstance(local, dict) else ""
    if val:
        return val
    raise RuntimeError(
        f"no base URL configured for llm_provider={provider!r}. Set ${env_var} "
        f"or artifact.local.json's llm_base_urls.{provider} "
        f"(see artifact.local.example.json)."
    )


def _build_llm(config: RefineConfig) -> LLM:
    """Build the refinement LLM from ``config.model`` / ``config.llm_provider``."""
    if config.llm_provider not in _PROVIDER_BASE_URL_ENV_VARS:
        valid = ", ".join(sorted(_PROVIDER_BASE_URL_ENV_VARS))
        raise ValueError(f"unknown llm_provider {config.llm_provider!r}; valid: {valid}")
    base_url = _resolve_provider_base_url(config.llm_provider)
    api_key = ""
    for var in _API_KEY_ENV_CANDIDATES:
        val = os.environ.get(var, "").strip()
        if val:
            api_key = val
            break
    if not api_key:
        candidates = ", ".join(f"${v}" for v in _API_KEY_ENV_CANDIDATES)
        raise RuntimeError(f"no API key found; set one of {candidates}")
    return LLM(model=config.model, api_key=SecretStr(api_key), base_url=base_url, drop_params=True)


# ---------------------------------------------------------------------------
# Summarizer dispatch. "none" is the w/o-Summary ablation: the summary stage is
# removed and the raw trace text is passed through to embedding, clustering and
# proposals unchanged. Anything else fails loudly.
# ---------------------------------------------------------------------------


def _make_summarizer(name: str, polarity: str, skill_name: str | None) -> TraceSummarizer:
    """Build the polarity-steered summarizer for ``name`` ("full_trace" or "none")."""
    if name == "none":
        # w/o-Summary ablation: the raw trace text stands in for the summary.
        return NoSummaryTraceSummarizer(polarity=polarity, skill_name=skill_name)
    if name == "full_trace":
        return LLMTraceSummarizer(polarity=polarity, skill_name=skill_name)
    raise ValueError(
        f"unknown summarizer {name!r}; this artifact ships only 'full_trace' and 'none'"
    )


def _append_ground_truth_event(trace: Trace, gt_text: str) -> Trace:
    """Return a copy of ``trace`` with a trailing SCORING REFERENCE span.

    The reference states the expected ground-truth answer next to the agent's
    output so a GT-aware summarizer can diagnose over- vs under-decoration of the
    final answer (e.g. `108°` emitted when `108` was expected) instead of guessing
    the failure direction blind. Placed last so it travels with the final `Answer:`
    line even if the trace is chunked. Labeled as NOT part of the agent's output.
    """
    note = (
        "SCORING REFERENCE (metadata, NOT part of the agent's output): the hidden "
        "grader compares this run's final `Answer:` line against this exact expected "
        "ground-truth answer after light normalization.\n"
        f"EXPECTED GROUND-TRUTH ANSWER: {gt_text}"
    )
    return Trace(list(trace) + [TraceEvent({"kind": "raw_span", "output": note})])


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

def _load_binary_scores(binary_rewards: Path | None) -> dict[str, bool] | None:
    """{trace_id: passed} from binary_rewards.json, or None if the file is missing."""
    if not (binary_rewards and binary_rewards.exists()):
        return None
    data = json.loads(binary_rewards.read_text(encoding="utf-8"))
    return {tid: bool(v) for tid, v in data.items()}


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def _trace_dir_name(trace_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_id).strip("._-") or "trace"
    digest = hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:12]
    if len(sanitized) > 80:
        sanitized = sanitized[:80].rstrip("._-") or "trace"
    return f"trace_{sanitized}_{digest}"


def _jsonl_trace_to_trace(raw: dict) -> Trace:
    events = []
    for span in raw.get("spans", []):
        output = span.get("output_text", "") or ""
        if not output:
            continue
        events.append(
            TraceEvent(
                {
                    "kind": "raw_span",
                    "span_id": span.get("span_id", ""),
                    "name": span.get("name", ""),
                    "input": span.get("input_text", "") or "",
                    "output": output,
                    "start_time": span.get("start_time", ""),
                    "end_time": span.get("end_time", ""),
                }
            )
        )
    return Trace(events)


def _load_traces(jsonl_path: Path, limit: int | None = None) -> list[tuple[str, Trace]]:
    traces: list[tuple[str, Trace]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            trace_id = raw.get("trace_id", "")
            if not trace_id:
                continue
            if not any((span.get("output_text", "") or "") for span in raw.get("spans", [])):
                continue
            traces.append((trace_id, _jsonl_trace_to_trace(raw)))
            if limit is not None and len(traces) >= limit:
                break
    return traces


# ---------------------------------------------------------------------------
# Summary cache
# ---------------------------------------------------------------------------


def _summary_cache_path(cache_dir: Path, trace_id: str) -> Path:
    return cache_dir / _trace_dir_name(trace_id) / "summary.json"


def _load_cached_summary(
    cache_path: Path,
    expected_trace_id: str,
    expected_technique: str,
) -> TraceSummary | None:
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("trace_id") != expected_trace_id:
        return None
    cached_technique = data.get("technique", "")
    if cached_technique and cached_technique != expected_technique:
        return None
    summary = data.get("summary", "")
    observations = data.get("observations", [])
    if not isinstance(summary, str) or not isinstance(observations, list):
        return None
    return TraceSummary(
        trace_id=expected_trace_id,
        summary=summary,
        observations=tuple(item for item in observations if isinstance(item, str)),
        technique=cached_technique or expected_technique,
    )


def _save_summary(cache_dir: Path, summary: TraceSummary) -> None:
    td = cache_dir / _trace_dir_name(summary.trace_id)
    td.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": summary.trace_id,
        "summary": summary.summary,
        "observations": list(summary.observations),
        "technique": summary.technique,
    }
    (td / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if summary.prompt:
        (td / "summarizer_prompt.txt").write_text(summary.prompt, encoding="utf-8")


# ---------------------------------------------------------------------------
# Cluster + refine for one partition
# ---------------------------------------------------------------------------


async def _run_partition(
    *,
    polarity: str,
    summaries: list[TraceSummary],
    skill: Skill,
    llm: LLM,
    output_dir: Path,
    embedding_base_url: str,
    embedding_model: str,
    embedding_api_key: str,
    method: str,
    summarizer_name: str,
    jsonl_path: Path,
    limit: int | None,
    binary_scores: dict[str, bool] | None,
    skill_name: str | None = None,
    verify_negative: bool = False,
    min_cluster_size: int | None = None,
) -> list[ClusterProposal]:
    """Cluster one outcome partition and propose one edit per cluster (paper §2.3–2.5).

    Returns the partition's surviving cluster proposals; ``refine()`` merges both
    partitions' proposals into the skill via ``merge_polarity_proposals``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{polarity}] {len(summaries)} summaries → clustering ({method})...")

    cluster_refiner = PolarityClusterRefiner(polarity, verify=verify_negative)
    refiner = ClusteredTraceDrivenRefiner(
        cluster_refiner=cluster_refiner,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        method=method,
        # Embeddings are cached under .cache/skill_refiner/summary_clustering, keyed
        # by the exact text + model, so they are safely reused across runs.
        clusterer=(
            UmapHdbscanClusterer(hdbscan_min_cluster_size=min_cluster_size)
            if min_cluster_size is not None and method == "umap_hdbscan"
            else None
        ),
    )

    ctx = RefinementContext(llm=llm)
    cluster_proposals = await refiner.propose_from_summaries(skill, summaries, ctx)
    print(f"  [{polarity}] {len(cluster_proposals)} cluster proposal(s)")

    # Persist what the evidence gate REJECTED, not just what survived. Without this
    # the gate is unauditable after the fact: cluster_proposals.json holds only
    # survivors and the run log keeps a truncated reason with no theme, so a dropped
    # proposal cannot be quoted, counted, or re-examined once the run ends.
    if cluster_refiner.rejected:
        (output_dir / "rejected_cluster_proposals.json").write_text(
            json.dumps(cluster_refiner.rejected, indent=2), encoding="utf-8"
        )
        print(f"  [{polarity}] {len(cluster_refiner.rejected)} proposal(s) dropped by the "
              f"evidence gate → rejected_cluster_proposals.json")

    (output_dir / "cluster_proposals.json").write_text(
        json.dumps(
            [
                {
                    "cluster_id": p.cluster_id,
                    "theme": p.theme,
                    "reinforce": list(p.reinforce),
                    "soften": list(p.soften),
                    "suggested_edit": p.suggested_edit,
                    "n_traces": p.n_traces,
                    "cluster_size": p.cluster_size,
                    "polarity": p.polarity,
                }
                for p in cluster_proposals
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    score_counts = {}
    if binary_scores:
        pos = sum(1 for s in summaries if binary_scores.get(s.trace_id) is True)
        neg = sum(1 for s in summaries if binary_scores.get(s.trace_id) is False)
        score_counts = {"n_positive": pos, "n_negative": neg}

    manifest = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "summarizer": summarizer_name,
        "polarity": polarity,
        "skill_name": skill_name,
        "model": getattr(llm, "model", "") or "",
        "base_url": getattr(llm, "base_url", "") or "",
        "skill_file": str(skill.name),
        "jsonl": str(jsonl_path),
        "limit": limit,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "clustering_method": method,
        "n_summaries": len(summaries),
        "n_cluster_proposals": len(cluster_proposals),
        "n_rejected_by_gate": len(cluster_refiner.rejected),
        "trace_ids": [s.trace_id for s in summaries],
        **score_counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (output_dir / "summaries.json").write_text(
        json.dumps(
            [
                {
                    "trace_id": s.trace_id,
                    "summary": s.summary,
                    "observations": list(s.observations),
                    "binary_score": binary_scores.get(s.trace_id) if binary_scores else None,
                }
                for s in summaries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    for p in cluster_proposals:
        tag = "noise" if p.cluster_id < 0 else f"cluster {p.cluster_id}"
        print(f"    [{tag}] {p.theme} ({p.n_traces} traces)")

    return cluster_proposals


def split_trace_groups(
    all_traces: list[tuple[str, Trace]],
    binary_scores: dict[str, bool] | None,
    *,
    without_positive: bool = False,
    without_negative: bool = False,
) -> dict[str, list[tuple[str, Trace]]]:
    """Partition traces by binary score, before any summarization happens.

    With scores: {"positive", "negative"}; traces carrying no score belong to
    neither and are dropped with a warning. Without scores: a single "all" group.

    ``without_positive`` / ``without_negative`` are the two single-polarity
    ablations. The named partition is removed at the earliest point it exists, so
    none of its traces is ever summarized, embedded, clustered or proposed from.
    The surviving partition is returned UNCHANGED -- the two are independent
    (separate summary caches, separate clusterings, separate refiners).

    Traces carrying no score stay in neither partition either way: "no positives"
    must not quietly become "everything that is not a positive", and vice versa.

    Raises ValueError if an ablation is requested without a polarity split (there
    is then no component to remove, and the run would quietly produce the ordinary
    'all' partition under an ablation's name), or if both are requested at once
    (that removes the entire corpus, not one component).
    """
    if without_positive and without_negative:
        raise ValueError(
            "--without-positive and --without-negative are mutually exclusive; "
            "together they remove every trace, which is not an ablation of one "
            "component."
        )
    if not binary_scores:
        if without_positive or without_negative:
            flag = "--without-positive" if without_positive else "--without-negative"
            side = "positive" if without_positive else "negative"
            raise ValueError(
                f"{flag} needs a polarity split; pass --binary-rewards so the "
                f"{side} partition exists to be removed."
            )
        return {"all": all_traces}

    pos = [(tid, t) for tid, t in all_traces if binary_scores.get(tid) is True]
    neg = [(tid, t) for tid, t in all_traces if binary_scores.get(tid) is False]
    unscored = [(tid, t) for tid, t in all_traces if tid not in binary_scores]
    print(f"  positive: {len(pos)}, negative: {len(neg)}, unscored: {len(unscored)}")
    if unscored:
        print(f"  WARNING: {len(unscored)} traces have no score — excluded from both partitions")

    if without_positive:
        print(
            f"  [w/o positive] dropped all {len(pos)} positive traces; "
            f"{len(neg)} negative traces retained"
        )
        return {"negative": neg}
    if without_negative:
        # NOTE the verify gate goes with them. PolarityClusterRefiner sets
        # `self._verify = verify and polarity == "negative"`, so --verify-negative-
        # clusters has no consumer once this partition is gone. The flag stays ON and
        # the config is unchanged; it simply executes zero times. That is a
        # consequence of removing negatives, not a second ablation -- but it is the
        # reason this condition cannot claim "gating held constant" operatively.
        print(
            f"  [w/o negative] dropped all {len(neg)} negative traces; "
            f"{len(pos)} positive traces retained "
            f"(the negative-only verify gate has no input and will not run)"
        )
        return {"positive": pos}
    return {"positive": pos, "negative": neg}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


async def refine(config: RefineConfig) -> RefinementProposal | None:
    """Run the end-to-end refine loop for one config: summarize, cluster and
    propose per polarity partition, then merge both partitions into one
    proposed skill via ``merge_polarity_proposals``.

    Returns the merged proposal, or None if synthesis had nothing to propose.
    """
    llm = _build_llm(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Summarizer        : {config.summarizer}")
    print(f"Positive partition: {'on' if config.use_positive else 'REMOVED (use_positive=False)'}")
    print(f"Negative partition: {'on' if config.use_negative else 'REMOVED (use_negative=False)'}")
    print(f"Model             : {config.model}")
    print(f"Skill file        : {config.skill_file}")
    print(f"Skill name        : {config.skill_name or '(unset)'}")
    print(f"JSONL             : {config.raw_jsonl}")
    print(f"Embedding server  : {config.embedding_base_url}  model={config.embedding_model}")
    print(f"Clustering method : {config.cluster_method}")

    # The two stages load the SAME file two different ways, exactly as the two
    # source scripts did — this is not an oversight to tidy away.
    #
    # Refine (run_summarizer_ablation.py:904) used Skill.load, which parses the
    # YAML frontmatter off into fields and rewrites the body's header. The
    # per-cluster proposal prompts only quote the skill for context, so the
    # stripped form is what the 81.0% run's refine stage saw.
    skill = Skill.load(config.skill_file, strict=False)
    # Merge (merge_signal_passes.py:123) built the Skill from the RAW file text:
    # `Skill(name=base_skill.stem, content=base_skill.read_text(...))`. That
    # matters because the merge prompt embeds `skill.content` verbatim as
    # "Current skill content" (propose/clustered.py) and the LLM's reply is
    # written straight out as combined/proposed_skill.md. On the record's own
    # base, base_true.md, Skill.load drops 312 bytes of frontmatter — including
    # the `description:` line the SpreadsheetBench harness triggers the skill on.
    # The record's output, minclust2_soften.md, still carries that frontmatter,
    # which is only possible because its merge input did.
    merge_skill = Skill(
        name=config.skill_file.stem,
        content=config.skill_file.read_text(encoding="utf-8"),
    )

    # ------------------------------------------------------------------ scores
    binary_scores = _load_binary_scores(config.binary_rewards)
    if binary_scores:
        pos = sum(1 for v in binary_scores.values() if v)
        neg = sum(1 for v in binary_scores.values() if not v)
        print(f"Binary scores     : {len(binary_scores)} traces — {pos} positive, {neg} negative")
    else:
        print("Binary scores     : none")

    # ------------------------------------------------------------------ traces
    print("\nLoading traces...")
    all_traces = _load_traces(config.raw_jsonl, limit=config.limit)
    print(f"  {len(all_traces)} traces loaded (limit={config.limit})")

    if config.ground_truth_file is not None:
        gt_raw = json.loads(config.ground_truth_file.read_text(encoding="utf-8"))
        if not isinstance(gt_raw, dict) or not gt_raw:
            raise ValueError(
                "ground_truth_file must be a non-empty {trace_id: answer} object: "
                f"{config.ground_truth_file}"
            )
        ground_truth = {str(k): str(v) for k, v in gt_raw.items()}
        matched = sum(1 for tid, _ in all_traces if tid in ground_truth)
        all_traces = [
            (tid, _append_ground_truth_event(t, ground_truth[tid]) if tid in ground_truth else t)
            for tid, t in all_traces
        ]
        print(f"  Injected ground-truth SCORING REFERENCE into {matched}/{len(all_traces)} traces")

    # Split traces by score before summarizing so each group is fully independent.
    trace_groups = split_trace_groups(
        all_traces, binary_scores,
        without_positive=not config.use_positive,
        without_negative=not config.use_negative,
    )
    if "positive" not in trace_groups and "negative" not in trace_groups:
        # binary_rewards was missing/empty, so split_trace_groups fell back to the
        # polarity-blind "all" partition. merge_polarity_proposals has no input for
        # that case (it takes positive/negative cluster lists, not a neutral one),
        # and the configuration of record always has a positive/negative split — so
        # this is a data problem to surface, not a case to route around silently.
        raise RuntimeError(
            f"no positive/negative split from {config.binary_rewards}; "
            "refine() requires binary_rewards to resolve to a non-empty "
            "{trace_id: bool} mapping"
        )

    # ------------------------------------------------------------------ summarise
    eval_ctx = EvaluationContext(llm=llm, secrets={})

    # Per-polarity summary caches — polarity-specialized prompts produce
    # different summaries for the same trace, so caches must not be shared
    # across partitions.
    summaries_root = config.output_dir / "summaries"

    async def _summarize_group(
        polarity: str,
        traces: list[tuple[str, Trace]],
    ) -> list[TraceSummary]:
        summarizer = _make_summarizer(config.summarizer, polarity, config.skill_name)
        runner = TraceSummarizationRunner(summarizer=summarizer)
        technique = runner.technique
        summary_cache_dir = summaries_root / polarity
        summary_cache_dir.mkdir(parents=True, exist_ok=True)

        # Write raw content for inspection
        for trace_id, trace in traces:
            td = summary_cache_dir / _trace_dir_name(trace_id)
            td.mkdir(parents=True, exist_ok=True)
            raw_text = serialize_trace(trace)
            raw_path = td / "raw_content.txt"
            if not raw_path.exists():
                if raw_text.strip():
                    raw_path.write_text(raw_text, encoding="utf-8")
                else:
                    (td / "skipped.txt").write_text(
                        "No raw content after filtering derived event kinds.\n",
                        encoding="utf-8",
                    )

        cached: list[TraceSummary] = []
        uncached: list[tuple[str, Trace]] = []
        for trace_id, trace in traces:
            hit = _load_cached_summary(
                _summary_cache_path(summary_cache_dir, trace_id),
                trace_id,
                technique,
            )
            if hit is not None:
                cached.append(hit)
            else:
                uncached.append((trace_id, trace))

        print(
            f"\n[{polarity}] Summarizing {len(uncached)} traces "
            f"[{len(cached)} cached, technique={technique}]..."
        )
        new_summaries = await runner.summarize(uncached, eval_ctx)
        for summary in new_summaries:
            _save_summary(summary_cache_dir, summary)

        group_summaries = cached + new_summaries
        print(f"  [{polarity}] {len(group_summaries)} summaries ready")
        return group_summaries

    # ------------------------------------------------------------------ partition
    partitions: dict[str, list[TraceSummary]] = {}
    for group_name, traces in trace_groups.items():
        summaries = await _summarize_group(group_name, traces)
        if len(summaries) < 3:
            print(
                f"  [{group_name}] Only {len(summaries)} summaries — need ≥3, skipping.",
                file=sys.stderr,
            )
            continue
        partitions[group_name] = summaries

    if not partitions:
        raise RuntimeError("No partition has ≥3 summaries. Try a larger config.limit.")

    # ------------------------------------------------------------------ cluster + refine
    cluster_proposals_by_polarity: dict[str, list[ClusterProposal]] = {
        "positive": [], "negative": [],
    }
    for polarity, summaries in partitions.items():
        cluster_proposals_by_polarity[polarity] = await _run_partition(
            polarity=polarity,
            summaries=summaries,
            skill=skill,
            llm=llm,
            output_dir=config.output_dir / polarity,
            embedding_base_url=config.embedding_base_url,
            embedding_model=config.embedding_model,
            embedding_api_key=os.environ.get("EMBEDDING_API_KEY", ""),
            method=config.cluster_method,
            summarizer_name=config.summarizer,
            jsonl_path=config.raw_jsonl,
            limit=config.limit,
            binary_scores=binary_scores,
            skill_name=config.skill_name,
            verify_negative=config.verify_negative_clusters,
            min_cluster_size=config.min_cluster_size,
        )

    # ------------------------------------------------------------------ cross-polarity merge
    # Mirrors merge_signal_passes.py's _merge_variant/_load: the HDBSCAN noise
    # bucket (cluster_id < 0) is excluded, and negative cluster ids are offset by
    # 100 so they cannot collide with positive ids from the independent clustering
    # pass once both lists are shown to the LLM together.
    positive_clusters = [p for p in cluster_proposals_by_polarity["positive"] if p.cluster_id >= 0]
    negative_clusters = [
        replace(p, cluster_id=100 + p.cluster_id)
        for p in cluster_proposals_by_polarity["negative"] if p.cluster_id >= 0
    ]

    if not positive_clusters and not negative_clusters:
        print("No cluster proposals survived on either polarity — nothing to merge.",
              file=sys.stderr)
        return None

    total_traces = sum(p.cluster_size for p in positive_clusters + negative_clusters)
    print(
        f"\n[combined] merging {len(positive_clusters)} positive + "
        f"{len(negative_clusters)} negative cluster(s) over {total_traces} traces..."
    )
    merge_ctx = RefinementContext(llm=llm)
    proposal = await merge_polarity_proposals(
        positive_clusters, negative_clusters, merge_skill, merge_ctx,
        total_traces=total_traces,
    )

    if proposal is None:
        print("[combined] merge returned no proposal.", file=sys.stderr)
    else:
        combined_dir = config.output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        (combined_dir / "proposed_skill.md").write_text(
            proposal.proposed_content, encoding="utf-8"
        )
        (combined_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "merge": "polarity_aware (merge_polarity_proposals, noise excluded)",
                    "model": config.model,
                    "skill_file": str(config.skill_file),
                    "n_positive_clusters": len(positive_clusters),
                    "n_negative_clusters": len(negative_clusters),
                    "total_traces": total_traces,
                    "confidence": proposal.confidence,
                    "rationale": proposal.rationale,
                    "observations": list(proposal.observations),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[combined] → {combined_dir / 'proposed_skill.md'}")

    # ------------------------------------------------------------------ token usage
    m = llm.metrics
    tu = m.accumulated_token_usage
    print("\nToken usage:")
    if m.accumulated_cost:
        print(f"  Cost              : ${m.accumulated_cost:.4f}")
    if tu:
        print(f"  Prompt tokens     : {tu.prompt_tokens:,}")
        print(f"  Completion tokens : {tu.completion_tokens:,}")
    print(f"  LLM calls made    : {len(m.costs)}")
    print(f"\nOutput root: {config.output_dir}")

    (config.output_dir / "usage.json").write_text(
        json.dumps(
            {
                "model": config.model,
                "cost": float(m.accumulated_cost or 0.0),
                "prompt_tokens": tu.prompt_tokens if tu else 0,
                "completion_tokens": tu.completion_tokens if tu else 0,
                "llm_calls": len(m.costs),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return proposal

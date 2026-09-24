from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

# Resolved relative to this file (repo root / .cache / ...) rather than the
# process's current working directory, so the default doesn't silently write a
# .cache/ directory wherever the caller happened to invoke the script from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "skill_refiner" / "summary_clustering"

# Embedding-stage resilience: retry each embedding call before giving up.
_EMBED_ATTEMPTS = 4
_EMBED_BACKOFF = 1.0             # seconds, doubled per attempt

_CACHE_LOCKS: dict[Path, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock_for(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _CACHE_LOCKS_GUARD:
        if resolved not in _CACHE_LOCKS:
            _CACHE_LOCKS[resolved] = threading.Lock()
        return _CACHE_LOCKS[resolved]


class EmbeddingClient:
    def __init__(self, *, base_url: str, model: str, api_key: str = "") -> None:
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key or "local", base_url=base_url or None)

    def embed_one(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


def _embed_cache_path(text: str, model: str, base_url: str, *, cache_dir: Path) -> Path:
    key = hashlib.sha256(
        json.dumps(
            {"text": text, "model": model, "base_url": base_url},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.json"


def _load_cached_embedding(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        return None
    return [float(v) for v in vector]


def _write_cached_embedding(path: Path, vector: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({"embedding": vector}, separators=(",", ":"))
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def embed_items(
    items: list,
    extractor,
    *,
    client: EmbeddingClient,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    max_workers: int = 4,
) -> dict[str, list[float]]:
    """Return {trace_id -> vector} for items where extractor.embed_text is non-empty."""
    vectors: dict[str, list[float]] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def _embed(item) -> None:
        text = extractor.embed_text(item)
        tid = extractor.trace_id(item)
        if not text:
            return
        cache_path = _embed_cache_path(text, client.model, client.base_url, cache_dir=cache_dir)
        cached = _load_cached_embedding(cache_path)
        if cached is not None:
            with lock:
                vectors[tid] = cached
            return
        vector = None
        for attempt in range(_EMBED_ATTEMPTS):
            try:
                vector = client.embed_one(text)
                break
            except Exception as exc:  # noqa: PERF203
                if attempt == _EMBED_ATTEMPTS - 1:
                    with lock:
                        errors.append(f"{tid}: {type(exc).__name__}: {exc}")
                    return
                time.sleep(_EMBED_BACKOFF * (2**attempt))
        if vector is None:
            return
        with _cache_lock_for(cache_path):
            _write_cached_embedding(cache_path, vector)
        with lock:
            vectors[tid] = vector

    worker_count = min(max(max_workers, 1), max(len(items), 1))
    if items:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_embed, item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if errors and not vectors:
        raise RuntimeError(
            f"All {len(errors)} embedding call(s) failed. "
            f"Is the embedding server running?\nFirst error: {errors[0]}"
        )
    # Partial failures are an error: an item with no vector gets no cluster
    # assignment, so group_items_by_cluster would skip it and the run would
    # silently cover fewer traces than it was given.
    expected = sum(1 for item in items if extractor.embed_text(item))
    if len(vectors) < expected:
        missing = expected - len(vectors)
        detail = f"\nFirst error: {errors[0]}" if errors else ""
        raise RuntimeError(
            f"Embedding produced vectors for only {len(vectors)} of {expected} items "
            f"({missing} missing after {_EMBED_ATTEMPTS} attempts each). Refusing to "
            "continue: the missing items would be silently dropped from clustering "
            f"and this run would not be comparable to others.{detail}"
        )
    # Determinism. The ThreadPoolExecutor above fills `vectors` in COMPLETION order,
    # and that insertion order becomes the ROW order of the matrix built by
    # cluster_by_umap_hdbscan (`ids = list(vectors_by_id.keys())`). UMAP's
    # random_state fixes the embedding for a GIVEN row order but does not make it
    # invariant to permuting the rows, so the same corpus could cluster differently
    # from run to run. Rebuild in the caller's item order so the matrix is
    # reproducible.
    return {
        tid: vectors[tid]
        for tid in (extractor.trace_id(item) for item in items)
        if tid in vectors
    }


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def cluster_by_umap_hdbscan(
    vectors_by_id: dict[str, list[float]],
    *,
    umap_n_components: int = 20,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 2,
    hdbscan_min_samples: int = 1,
    random_state: int = 42,
) -> dict[str, int]:
    """UMAP dimensionality reduction followed by HDBSCAN clustering."""
    if not vectors_by_id:
        return {}
    try:
        import hdbscan as hdbscan_lib
        import numpy as np
        import umap as umap_lib
    except ImportError as exc:
        raise ImportError(f"Missing dependency: {exc}. Install umap-learn and hdbscan.") from exc

    ids = list(vectors_by_id.keys())
    X = np.array([_l2_normalize(vectors_by_id[cid]) for cid in ids])

    n_neighbors = min(umap_n_neighbors, len(ids) - 1) if len(ids) > 1 else 1
    # UMAP spectral init requires n_components < n_samples - 1; cap at n-2
    n_components = min(umap_n_components, max(1, len(ids) - 2)) if len(ids) > 2 else 1

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
    )
    X_reduced = reducer.fit_transform(X)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_reduced)

    group_by_id: dict[str, int] = {}
    noise_counter = -1
    for cid, label in zip(ids, labels, strict=True):
        if int(label) == -1:
            group_by_id[cid] = noise_counter
            noise_counter -= 1
        else:
            group_by_id[cid] = int(label)
    return group_by_id


def cluster_single(vectors: dict[str, list[float]]) -> dict[str, int]:
    """Assign every trace to one cluster (cluster 0). Disables clustering —
    the refiner then does a single flat pass over all summaries."""
    return {tid: 0 for tid in vectors}


def group_items_by_cluster(
    items: list,
    assignments: dict[str, int],
    extractor,
) -> dict[int, list]:
    """Bucket items by cluster_id. All noise points (negative ids) fold into key -1.

    Works with any item type via a TextExtractor.
    """
    groups: dict[int, list] = {}
    for item in items:
        tid = extractor.trace_id(item)
        gid = assignments.get(tid)
        if gid is None:
            continue
        bucket = -1 if gid < 0 else gid
        groups.setdefault(bucket, []).append(item)
    return groups

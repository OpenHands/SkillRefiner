import inspect

from skill_refiner.cluster import cluster_by_umap_hdbscan


def test_min_cluster_size_defaults_to_two():
    """The configuration of record. At 3, most negatives fall below threshold."""
    sig = inspect.signature(cluster_by_umap_hdbscan)
    assert sig.parameters["hdbscan_min_cluster_size"].default == 2


def test_umap_defaults_match_the_configuration_of_record():
    p = inspect.signature(cluster_by_umap_hdbscan).parameters
    assert p["umap_n_components"].default == 20
    assert p["umap_n_neighbors"].default == 15
    assert p["umap_min_dist"].default == 0.1
    assert p["random_state"].default == 42
    assert p["hdbscan_min_samples"].default == 1


def test_empty_input_returns_empty_mapping():
    assert cluster_by_umap_hdbscan({}) == {}


def test_min_cluster_size_two_actually_reaches_hdbscan(monkeypatch):
    """The knob must reach HDBSCAN (and UMAP), not merely sit in the signature.

    inspect.signature only proves the default VALUE is 2; it would still pass
    if the function body ignored the parameter and hardcoded 3 internally.
    This spies on the real umap.UMAP and hdbscan.HDBSCAN constructors to prove
    the configured values are actually forwarded into the clustering call.
    """
    import hdbscan as hdbscan_lib
    import numpy as np
    import umap as umap_lib

    umap_captured: dict = {}
    hdbscan_captured: dict = {}

    class _SpyUMAP:
        def __init__(self, **kwargs):
            umap_captured.update(kwargs)

        def fit_transform(self, X):
            return np.asarray(X)

    class _SpyHDBSCAN:
        def __init__(self, **kwargs):
            hdbscan_captured.update(kwargs)

        def fit_predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(umap_lib, "UMAP", _SpyUMAP)
    monkeypatch.setattr(hdbscan_lib, "HDBSCAN", _SpyHDBSCAN)

    # 30 traces of 5-dim vectors so the n_components/n_neighbors caps
    # (min(default, len(ids) - k)) never bind -- the values captured below are
    # therefore the literal defaults, not a size-driven coincidence.
    vectors = {
        f"t{i}": [float(i), float(i + 1), float(i + 2), float(i + 3), float(i + 4)]
        for i in range(30)
    }
    cluster_by_umap_hdbscan(vectors)

    assert hdbscan_captured["min_cluster_size"] == 2
    assert hdbscan_captured["min_samples"] == 1
    assert umap_captured["n_components"] == 20
    assert umap_captured["n_neighbors"] == 15
    assert umap_captured["min_dist"] == 0.1
    assert umap_captured["random_state"] == 42

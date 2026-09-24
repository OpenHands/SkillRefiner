import pytest

from skill_refiner.propose import ClusteredTraceDrivenRefiner
from skill_refiner.propose.clustered import SingleClusterer, UmapHdbscanClusterer


def test_unknown_clustering_method_raises():
    """A mistyped or deleted method must fail loudly, not silently default."""
    with pytest.raises(ValueError, match="unknown clustering method"):
        ClusteredTraceDrivenRefiner(cluster_refiner=None, method="dbscan")


def test_umap_hdbscan_method_still_constructs():
    refiner = ClusteredTraceDrivenRefiner(cluster_refiner=None, method="umap_hdbscan")
    assert isinstance(refiner._clusterer, UmapHdbscanClusterer)


def test_single_method_still_constructs():
    refiner = ClusteredTraceDrivenRefiner(cluster_refiner=None, method="single")
    assert isinstance(refiner._clusterer, SingleClusterer)

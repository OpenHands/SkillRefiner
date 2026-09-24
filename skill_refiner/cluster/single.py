"""Single-cluster strategy — the faithful no-clustering ablation.

Returns one cluster containing every item, so each polarity partition gets
exactly one refine pass. Changes nothing about the score-based partitioning.
"""

from skill_refiner.cluster.umap_hdbscan import cluster_single


class SingleClusterer:
    """Clusterer that puts every item in one cluster (method='single')."""

    def cluster(self, vectors: dict[str, list[float]]) -> dict[str, int]:
        return cluster_single(vectors)

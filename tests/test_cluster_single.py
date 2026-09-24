from skill_refiner.cluster import SingleClusterer


def test_every_item_lands_in_cluster_zero():
    vectors = {"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.5, 0.5]}
    assert SingleClusterer().cluster(vectors) == {"a": 0, "b": 0, "c": 0}


def test_empty_input_returns_empty_mapping():
    assert SingleClusterer().cluster({}) == {}

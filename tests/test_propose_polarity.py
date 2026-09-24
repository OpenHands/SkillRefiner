import pytest

from skill_refiner.propose import PolarityClusterRefiner


def test_positive_refiner_does_not_verify():
    """The gate is negative-only: positive clusters come from passing traces."""
    assert PolarityClusterRefiner("positive", verify=True)._verify is False


def test_negative_refiner_honours_the_verify_flag():
    assert PolarityClusterRefiner("negative", verify=True)._verify is True
    assert PolarityClusterRefiner("negative", verify=False)._verify is False


def test_unknown_polarity_is_rejected():
    with pytest.raises(ValueError):
        PolarityClusterRefiner("sideways")


def test_rejected_record_starts_empty():
    assert PolarityClusterRefiner("negative", verify=True).rejected == []

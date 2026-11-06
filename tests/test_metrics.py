"""Tests for src.metrics, using tiny hand-made examples."""

import pytest

from src.metrics import rmspe


def test_rmspe_perfect_prediction_is_zero():
    """A perfect forecast has no error."""
    assert rmspe([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_rmspe_hand_calculation():
    """Percentage errors of 100% and 0% give sqrt((1^2 + 0^2) / 2) = sqrt(0.5)."""
    assert rmspe([1.0, 4.0], [2.0, 4.0]) == pytest.approx(0.5**0.5)


def test_rmspe_errors_are_relative_to_y_true():
    """Errors of 0.1 on 1 and 0.2 on 2 are both 10%, so RMSPE is 0.1."""
    assert rmspe([1.0, 2.0], [1.1, 1.8]) == pytest.approx(0.1)


def test_rmspe_rejects_zero_in_y_true():
    """Dividing by a zero actual value is undefined."""
    with pytest.raises(ValueError):
        rmspe([0.0, 1.0], [0.5, 1.0])


def test_rmspe_rejects_shape_mismatch():
    """Inputs of different lengths must not be broadcast together."""
    with pytest.raises(ValueError):
        rmspe([1.0, 2.0], [1.0])

"""Tests for src.day_slots, using tiny hand-made examples."""

import numpy as np
import pytest

from src.day_slots import (
    assign_day_slots,
    boundary_scores,
    consecutive_distances,
    detect_boundary_windows,
    dominant_period,
    drop_false_boundaries,
    otsu_threshold,
    segment_length_counts,
)


def test_consecutive_distances_hand_calculation():
    """A 3-4-5 triangle gives a distance of 5, then a repeated row gives 0."""
    points = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    assert consecutive_distances(points).tolist() == pytest.approx([5.0, 0.0])


def test_dominant_period_finds_a_known_cycle():
    """A pure cycle of 13 steps is recovered."""
    series = np.sin(2 * np.pi * np.arange(1300) / 13)
    assert dominant_period(series) == pytest.approx(13.0)


def test_boundary_scores_add_both_sides():
    """Element t scores window t + 1 using the steps on either side of it."""
    assert boundary_scores(np.array([1.0, 2.0, 4.0])).tolist() == [3.0, 6.0]


def test_otsu_threshold_separates_two_groups():
    """The threshold falls between two well-separated groups."""
    values = np.r_[np.zeros(100), np.ones(50) * 10]
    assert 0 < otsu_threshold(values) < 10


def test_detect_boundary_windows_keeps_highest_score_in_each_run():
    """Scores at 1-2 form one run, and score index t maps to window t + 1."""
    scores = np.array([0.0, 5.0, 7.0, 0.0, 0.0, 6.0, 0.0])
    assert detect_boundary_windows(scores, 1.0).tolist() == [3, 6]


def test_segment_length_counts():
    """Gaps between boundaries are counted by length."""
    assert segment_length_counts(np.array([2, 15, 28, 40])) == {12: 1, 13: 2}


def test_drop_false_boundaries_removes_the_weak_split():
    """Segments of 4 and 9 fit in one day, so the weak boundary between them goes."""
    boundaries = np.array([12, 25, 29, 38, 51])
    scores = np.zeros(60)
    scores[boundaries - 1] = [5.0, 5.0, 1.0, 5.0, 5.0]
    assert drop_false_boundaries(boundaries, scores, max_length=14).tolist() == [12, 25, 38, 51]


def test_drop_false_boundaries_merges_a_split_at_the_start():
    """Windows 0-3 and 4-12 fit in one day, so the boundary at window 3 goes."""
    boundaries = np.array([3, 12, 25])
    scores = np.zeros(40)
    scores[boundaries - 1] = [1.0, 5.0, 5.0]
    assert drop_false_boundaries(boundaries, scores, max_length=14).tolist() == [12, 25]


def test_drop_false_boundaries_keeps_full_days():
    """Two 13-window days are too long to merge, so nothing is removed."""
    boundaries = np.array([12, 25, 38])
    assert drop_false_boundaries(boundaries, np.ones(50), max_length=14).tolist() == [12, 25, 38]


def test_assign_day_slots_counts_back_from_each_boundary():
    """Slot 0 is the boundary, and trailing windows are placed in a typical day."""
    day_id, slot = assign_day_slots(9, np.array([2, 5]), typical_length=3)
    assert day_id.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert slot.tolist() == [2, 1, 0, 2, 1, 0, 2, 1, 0]


def test_assign_day_slots_rejects_a_long_tail():
    """More than one day's windows after the last boundary means one was missed."""
    with pytest.raises(ValueError):
        assign_day_slots(10, np.array([1]), typical_length=3)

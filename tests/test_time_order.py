"""Tests for src.time_order, using tiny hand-made examples."""

import numpy as np
import polars as pl
import pytest

from src.time_order import (
    build_log_price_matrix,
    estimate_tick_sizes,
    lag_autocorrelation,
    nearest_neighbor_chain,
    pairwise_distances,
    path_length,
    two_opt,
)


def make_book(time_id, bid1, ask1, bid2, ask2):
    """Return a small book DataFrame for one window."""
    return pl.DataFrame(
        {
            "time_id": [time_id] * len(bid1),
            "bid_price1": bid1,
            "ask_price1": ask1,
            "bid_price2": bid2,
            "ask_price2": ask2,
        }
    )


def test_estimate_tick_sizes_uses_smallest_gap_per_window():
    """Window 5 has a smallest gap of 0.01; window 7 has a smallest gap of 0.02."""
    book = pl.concat(
        [
            make_book(7, [1.00], [1.02], [0.96], [1.06]),
            make_book(5, [1.00, 1.01], [1.03, 1.03], [0.97, 0.97], [1.05, 1.05]),
        ]
    )
    result = estimate_tick_sizes(book)
    assert result["time_id"].to_list() == [5, 7]
    assert result["tick_size"].to_list() == pytest.approx([0.01, 0.02])


def test_estimate_tick_sizes_ignores_noise_below_floor():
    """A gap below 1e-9 is noise, so the smallest real gap is 0.01."""
    book = make_book(1, [1.0], [1.0 + 1e-10], [0.99], [1.02])
    assert estimate_tick_sizes(book)["tick_size"][0] == pytest.approx(0.01)


def test_build_log_price_matrix_fills_missing_with_stock_median():
    """A tick of 0.01 is a $1 price, and stock 2 has one missing window."""
    ticks = pl.DataFrame(
        {
            "stock_id": [1, 1, 1, 2, 2],
            "time_id": [10, 20, 30, 10, 30],
            "tick_size": [0.01, 0.001, 0.0001, 0.01, 0.001],
        }
    )
    matrix, stock_ids, missing = build_log_price_matrix(ticks, [10, 20, 30])
    assert stock_ids.tolist() == [1, 2]
    assert matrix[:, 0] == pytest.approx([0.0, np.log(10), np.log(100)])
    assert missing.sum() == 1 and missing[1, 1]
    assert matrix[1, 1] == pytest.approx(np.log(10) / 2)


def test_nearest_neighbor_chain_follows_a_line():
    """Starting from one end of a line, the chain visits points in line order."""
    points = np.array([[0.0], [3.0], [1.0], [2.0]])
    order = nearest_neighbor_chain(pairwise_distances(points), start=0)
    assert order.tolist() == [0, 2, 3, 1]


def test_two_opt_untangles_a_line():
    """2-opt recovers the shortest path through points on a line, in either direction."""
    positions = np.arange(8.0)[:, None]
    distances = pairwise_distances(positions)
    order, _, converged = two_opt(distances, [0, 4, 3, 2, 1, 5, 7, 6])
    assert converged
    assert path_length(distances, order) == pytest.approx(7.0)
    assert sorted(order.tolist()) == list(range(8))


def test_lag_autocorrelation_hand_calculation():
    """For [1, 2, 3, 4], centered values give (0.75 - 0.25 + 0.75) / 5 = 0.25."""
    assert lag_autocorrelation([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)

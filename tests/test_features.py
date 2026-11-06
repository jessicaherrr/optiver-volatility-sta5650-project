"""Tests for src.features, using tiny hand-made examples."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_stock_features, realized_volatility, wap1


def test_wap1_hand_calculation():
    """(100*1 + 101*3) / (3+1) = 100.75: a large bid size pulls WAP toward the ask."""
    df = pd.DataFrame(
        {"bid_price1": [100.0], "ask_price1": [101.0], "bid_size1": [3], "ask_size1": [1]}
    )
    assert wap1(df).iloc[0] == pytest.approx(100.75)


def test_wap1_equal_sizes_gives_midpoint():
    """With equal sizes, WAP is the midpoint of the bid and ask."""
    df = pd.DataFrame(
        {"bid_price1": [10.0], "ask_price1": [12.0], "bid_size1": [5], "ask_size1": [5]}
    )
    assert wap1(df).iloc[0] == pytest.approx(11.0)


def test_realized_volatility_hand_calculation():
    """sqrt(3^2 + 4^2) = 5."""
    assert realized_volatility(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_realized_volatility_ignores_leading_nan():
    """The NaN that log_return leaves in the first row is skipped."""
    assert realized_volatility(pd.Series([np.nan, 3.0, 4.0])) == pytest.approx(5.0)


def make_book_rows(time_id, seconds, log_waps):
    """Return book rows whose level-1 WAP is exactly exp(log_wap) on each row."""
    wap = np.exp(log_waps)
    return pd.DataFrame(
        {
            "time_id": time_id,
            "seconds_in_bucket": seconds,
            "bid_price1": wap - 0.01,
            "ask_price1": wap + 0.01,
            "bid_price2": wap - 0.02,
            "ask_price2": wap + 0.02,
            "bid_size1": 2,
            "ask_size1": 2,
            "bid_size2": 3,
            "ask_size2": 1,
        }
    )


@pytest.fixture
def stock_features():
    """Return features for two windows; the book rows are deliberately unsorted."""
    book = pd.concat(
        [
            make_book_rows(7, [0, 10], [5.0, 5.2]),
            make_book_rows(5, [0, 400, 590], [0.0, 0.3, 0.7]),
        ]
    ).sample(frac=1, random_state=0)
    trade = pd.DataFrame(
        {
            "time_id": [5, 5],
            "seconds_in_bucket": [100, 200],
            "price": [1.0, 1.0],
            "size": [100, 50],
            "order_count": [2, 3],
        }
    )
    return build_stock_features(book, trade, stock_id=0).set_index("time_id")


def test_build_stock_features_one_row_per_time_id(stock_features):
    """Output is sorted by time_id, with one row each and the stock_id attached."""
    assert stock_features.index.tolist() == [5, 7]
    assert (stock_features["stock_id"] == 0).all()


def test_build_stock_features_realized_vol_windows(stock_features):
    """Window 5 has returns 0.3 at second 400 and 0.4 at second 590."""
    row = stock_features.loc[5]
    assert row["rv_full"] == pytest.approx(0.5)
    assert row["rv_last_60"] == pytest.approx(0.4)
    assert row["rv_last_120"] == pytest.approx(0.4)
    assert row["rv_last_300"] == pytest.approx(0.5)


def test_build_stock_features_no_return_across_windows(stock_features):
    """Window 7 only sees its own 0.2 return, not the jump from window 5's prices."""
    assert stock_features.loc[7, "rv_full"] == pytest.approx(0.2)


def test_build_stock_features_quiet_lookback_is_zero(stock_features):
    """A window with no book updates in the lookback has zero realized vol."""
    assert stock_features.loc[7, "rv_last_60"] == 0.0


def test_build_stock_features_book_summaries(stock_features):
    """Spread is 0.02, depth is 2+3+2+1 = 8, and imbalance is (5-3)/8 = 0.25."""
    row = stock_features.loc[5]
    assert row["mean_spread"] == pytest.approx(0.02)
    assert row["mean_depth"] == pytest.approx(8.0)
    assert row["mean_order_imbalance"] == pytest.approx(0.25)


def test_build_stock_features_trades(stock_features):
    """Window 5 has 150 shares over 5 orders; window 7 has no trades."""
    assert stock_features.loc[5, "trade_count"] == 5
    assert stock_features.loc[5, "avg_trade_size"] == pytest.approx(30.0)
    assert stock_features.loc[7, "trade_count"] == 0
    assert np.isnan(stock_features.loc[7, "avg_trade_size"])

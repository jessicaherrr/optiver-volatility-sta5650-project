"""Turn raw Optiver order book and trade data into per-window features."""

import numpy as np
import pandas as pd

WINDOW_SECONDS = 600
LOOKBACK_SECONDS = (60, 120, 300)


def wap1(df):
    """Return the weighted average price at book level 1.

    Formula: (bid_price1 * ask_size1 + ask_price1 * bid_size1)
    / (bid_size1 + ask_size1).

    Args:
        df: DataFrame with bid_price1, ask_price1, bid_size1, ask_size1.

    Returns:
        Series with one WAP per row.
    """
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


def wap2(df):
    """Return the weighted average price at book level 2.

    Formula: (bid_price2 * ask_size2 + ask_price2 * bid_size2)
    / (bid_size2 + ask_size2).

    Args:
        df: DataFrame with bid_price2, ask_price2, bid_size2, ask_size2.

    Returns:
        Series with one WAP per row.
    """
    return (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )


def log_return(series):
    """Return the one-step log returns of a price series.

    The first value is NaN because it has no earlier price. Call this once per
    time_id so that a return never crosses from one window into another.

    Args:
        series: Series of positive prices, in time order.

    Returns:
        Series of log returns with the same index as the input.
    """
    return np.log(series).diff()


def realized_volatility(returns):
    """Return realized volatility: the square root of the sum of squared returns.

    NaN values are skipped, so the leading NaN from log_return is safe to pass in.

    Args:
        returns: Series or array of log returns.

    Returns:
        Realized volatility as a float. This is 0.0 if there are no valid returns.
    """
    values = np.asarray(returns, dtype=float)
    return float(np.sqrt(np.nansum(np.square(values))))


def build_stock_features(book, trade, stock_id=None):
    """Build one row per time_id from one stock's book and trade data.

    All features describe the 600-second window. Output columns:

    - rv_full: realized volatility of WAP level 1 over the full window.
    - rv_last_60, rv_last_120, rv_last_300: the same, using only returns with
      seconds_in_bucket >= 600 - N. Returns are computed before filtering, so
      the first return in a lookback can start from an earlier book update.
      If the book did not update in the lookback, the value is 0.0.
    - mean_spread: mean of ask_price1 - bid_price1.
    - mean_order_imbalance: mean of (bid sizes - ask sizes) / (bid sizes + ask
      sizes), using levels 1 and 2. Ranges from -1 to 1; positive means more
      size on the bid side.
    - mean_depth: mean total size across levels 1 and 2 of both sides.
    - trade_count: total number of trade orders (sum of order_count). This is 0
      if the window has no trades.
    - avg_trade_size: shares per order (sum of size / sum of order_count). This
      is NaN if the window has no trades.

    Means are simple averages over book rows, not weighted by time. Trade rows
    are per-second totals, which is why the trade features use order_count
    rather than the number of rows. Every time_id in the book gets a row;
    time_ids that appear only in the trade data are dropped.

    Args:
        book: DataFrame with time_id, seconds_in_bucket, bid_price1, ask_price1,
            bid_price2, ask_price2, bid_size1, ask_size1, bid_size2, ask_size2.
        trade: DataFrame with time_id, seconds_in_bucket, price, size, order_count.
        stock_id: Optional stock ID to store in a stock_id column. Set this when
            reading one partition, because those files may not contain the column.

    Returns:
        DataFrame with one row per time_id, sorted by time_id.
    """
    book = book.sort_values(["time_id", "seconds_in_bucket"]).copy()
    book["log_ret"] = wap1(book).groupby(book["time_id"]).transform(log_return)

    bid_size = book["bid_size1"] + book["bid_size2"]
    ask_size = book["ask_size1"] + book["ask_size2"]
    book["spread"] = book["ask_price1"] - book["bid_price1"]
    book["imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)
    book["depth"] = bid_size + ask_size

    by_time = book.groupby("time_id")
    features = pd.DataFrame(
        {
            "rv_full": by_time["log_ret"].agg(realized_volatility),
            "mean_spread": by_time["spread"].mean(),
            "mean_order_imbalance": by_time["imbalance"].mean(),
            "mean_depth": by_time["depth"].mean(),
        }
    )

    for seconds in LOOKBACK_SECONDS:
        recent = book[book["seconds_in_bucket"] >= WINDOW_SECONDS - seconds]
        rv = recent.groupby("time_id")["log_ret"].agg(realized_volatility)
        features[f"rv_last_{seconds}"] = rv.reindex(features.index, fill_value=0.0)

    trades = trade.groupby("time_id")[["size", "order_count"]].sum()
    trades = trades.reindex(features.index, fill_value=0)
    features["trade_count"] = trades["order_count"].astype("int64")
    features["avg_trade_size"] = (trades["size"] / trades["order_count"]).where(
        trades["order_count"] > 0
    )

    columns = [
        "rv_full",
        "rv_last_60",
        "rv_last_120",
        "rv_last_300",
        "mean_spread",
        "mean_order_imbalance",
        "mean_depth",
        "trade_count",
        "avg_trade_size",
    ]
    features = features[columns].reset_index()
    if stock_id is not None:
        features.insert(0, "stock_id", stock_id)
    return features

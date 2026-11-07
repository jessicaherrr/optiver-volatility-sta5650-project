"""Recover the time order of the shuffled Optiver windows from price tick sizes.

Book prices are normalized within each window, but the smallest gap between
prices still reveals the tick size. Since US stocks trade in $0.01 ticks, the real
price level is about 0.01 / tick_size. Real prices drift slowly, so windows that
are close in time have similar price vectors across stocks.

Nothing in this module reads volatility or the target.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.manifold import SpectralEmbedding

PRICE_COLUMNS = ["bid_price1", "ask_price1", "bid_price2", "ask_price2"]
TICK_NOISE_FLOOR = 1e-9
US_TICK_DOLLARS = 0.01


def estimate_tick_sizes(book):
    """Estimate the tick size in each time window of one stock.

    For each time_id, this collects the distinct values of the four price columns,
    sorts them, and takes the smallest consecutive difference above 1e-9.

    Args:
        book: Polars DataFrame or LazyFrame with time_id and the four price columns.

    Returns:
        Polars DataFrame with columns time_id and tick_size, sorted by time_id.
        A window with no positive price difference has no row.
    """
    return (
        book.lazy()
        .select(
            pl.col("time_id"),
            pl.concat_list([pl.col(c).cast(pl.Float64) for c in PRICE_COLUMNS]).alias("price"),
        )
        .explode("price")
        .unique()
        .sort(["time_id", "price"])
        .with_columns(pl.col("price").diff().over("time_id").alias("gap"))
        .filter(pl.col("gap") > TICK_NOISE_FLOOR)
        .group_by("time_id")
        .agg(pl.col("gap").min().alias("tick_size"))
        .sort("time_id")
        .collect()
    )


def build_tick_table(book_dir):
    """Estimate tick sizes for every stock, reading one stock at a time.

    Args:
        book_dir: Path to book_train.parquet, partitioned as stock_id=<id>/.

    Returns:
        Polars DataFrame with columns stock_id, time_id, and tick_size.
    """
    frames = []
    stock_dirs = sorted(Path(book_dir).glob("stock_id=*"), key=lambda p: int(p.name.split("=")[1]))
    for stock_dir in stock_dirs:
        stock_id = int(stock_dir.name.split("=")[1])
        book = pl.scan_parquet(stock_dir / "*.parquet", hive_partitioning=False).select(
            ["time_id", *PRICE_COLUMNS]
        )
        frames.append(
            estimate_tick_sizes(book).with_columns(pl.lit(stock_id, dtype=pl.Int32).alias("stock_id"))
        )
    return pl.concat(frames).select(["stock_id", "time_id", "tick_size"])


def build_log_price_matrix(ticks, time_ids):
    """Build the time_id x stock matrix of log real prices.

    Each cell is log(0.01 / tick_size). Missing cells are filled with that stock's
    median log price.

    Args:
        ticks: Polars DataFrame with columns stock_id, time_id, and tick_size.
        time_ids: Sequence of all time_ids, which sets the row order.

    Returns:
        A tuple of (matrix, stock_ids, missing_mask): a float array with one row
        per time_id and one column per stock, the sorted stock IDs, and a boolean
        array marking the cells that were filled.
    """
    time_ids = np.asarray(time_ids)
    stock_ids = np.sort(ticks["stock_id"].unique().to_numpy())
    row = {time_id: i for i, time_id in enumerate(time_ids.tolist())}
    col = {stock_id: j for j, stock_id in enumerate(stock_ids.tolist())}

    matrix = np.full((len(time_ids), len(stock_ids)), np.nan)
    rows = np.array([row[t] for t in ticks["time_id"].to_list()])
    cols = np.array([col[s] for s in ticks["stock_id"].to_list()])
    matrix[rows, cols] = np.log(US_TICK_DOLLARS / ticks["tick_size"].to_numpy())

    missing_mask = np.isnan(matrix)
    medians = np.nanmedian(matrix, axis=0)
    matrix[missing_mask] = np.take(medians, np.nonzero(missing_mask)[1])
    return matrix, stock_ids, missing_mask


def standardize_columns(matrix):
    """Return the matrix with each column scaled to mean 0 and SD 1 (ddof=1)."""
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)


def pairwise_distances(points):
    """Return the matrix of Euclidean distances between rows of points."""
    squared = np.sum(points**2, axis=1)
    gram = points @ points.T
    return np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * gram, 0.0))


def spectral_order(points, random_state=0, n_neighbors=None):
    """Return an initial order from a 1-D spectral embedding of the rows.

    Args:
        points: Array with one row per window.
        random_state: Seed for the eigensolver.
        n_neighbors: Graph neighbors. None uses the scikit-learn default, which
            is 10% of the rows.

    Returns:
        Array of row indices sorted by the embedding value.
    """
    embedding = SpectralEmbedding(
        n_components=1,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state,
    ).fit_transform(points)
    return np.argsort(embedding[:, 0], kind="stable")


def nearest_neighbor_chain(distances, start):
    """Build an open path by always moving to the nearest unvisited point.

    Args:
        distances: Square distance matrix.
        start: Index of the first point.

    Returns:
        Array of point indices in visiting order.
    """
    n = distances.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    current = start
    for position in range(n):
        order[position] = current
        visited[current] = True
        if position < n - 1:
            candidates = np.where(visited, np.inf, distances[current])
            current = int(np.argmin(candidates))
    return order


def path_length(distances, order):
    """Return the total length of the open path that visits points in order."""
    order = np.asarray(order)
    return float(distances[order[:-1], order[1:]].sum())


def two_opt(distances, order, max_passes=500, tol=1e-9):
    """Shorten an open path by reversing segments until no reversal helps.

    Each pass tries every start position and applies the best reversal found
    there, including reversals that move either end of the path.

    Args:
        distances: Square distance matrix.
        order: Starting order of point indices.
        max_passes: Maximum number of full passes.
        tol: Minimum length reduction that counts as an improvement.

    Returns:
        A tuple of (order, passes, converged).
    """
    order = np.array(order, dtype=np.int64)
    n = len(order)
    for passes in range(1, max_passes + 1):
        improved = False

        # Reverse a prefix, which changes the start of the path.
        deltas = distances[order[0], order[1:]] - distances[order[:-1], order[1:]]
        j = int(np.argmin(deltas))
        if deltas[j] < -tol:
            order[: j + 1] = order[: j + 1][::-1]
            improved = True

        for i in range(n - 1):
            first, second = order[i], order[i + 1]
            removed = distances[first, second]

            # Reverse the suffix after i, which changes the end of the path.
            best_delta = distances[first, order[-1]] - removed
            best_j = n - 1

            # Reverse an interior segment, order[i + 1..j].
            if i + 2 < n - 1:
                ends = order[i + 2 : n - 1]
                nexts = order[i + 3 : n]
                deltas = distances[first, ends] + distances[second, nexts] - removed - distances[ends, nexts]
                k = int(np.argmin(deltas))
                if deltas[k] < best_delta:
                    best_delta = deltas[k]
                    best_j = i + 2 + k

            if best_delta < -tol:
                order[i + 1 : best_j + 1] = order[i + 1 : best_j + 1][::-1]
                improved = True

        if not improved:
            return order, passes, True
    return order, max_passes, False


def lag_autocorrelation(series, lag=1):
    """Return the sample autocorrelation at a lag, using the same formula as R's acf()."""
    values = np.asarray(series, dtype=float)
    centered = values - values.mean()
    return float(np.sum(centered[:-lag] * centered[lag:]) / np.sum(centered**2))

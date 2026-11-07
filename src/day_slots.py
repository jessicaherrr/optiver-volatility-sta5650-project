"""Find trading-day boundaries along the recovered time order, using prices only.

Prices jump overnight, so the distance between consecutive price vectors should
spike at each day boundary. The spikes usually come in adjacent pairs around one
window: a large move during that window, plus the overnight gap. For this reason,
each window is scored on the distances to both of its neighbors.

Nothing in this module reads volatility or the target.
"""

import numpy as np
from scipy.ndimage import median_filter


def consecutive_distances(points):
    """Return the Euclidean distance between each pair of consecutive rows."""
    return np.sqrt(np.sum((points[1:] - points[:-1]) ** 2, axis=1))


def dominant_period(series, min_period=2.0, max_period=100.0):
    """Return the strongest repeating period in a series, in steps.

    Args:
        series: Array of values in order.
        min_period: Shortest period to consider.
        max_period: Longest period to consider.

    Returns:
        The period with the largest periodogram value in the range.
    """
    values = np.asarray(series, dtype=float)
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    frequencies = np.fft.rfftfreq(len(values))
    with np.errstate(divide="ignore"):
        periods = 1.0 / frequencies
    in_range = (periods >= min_period) & (periods <= max_period)
    return float(periods[in_range][np.argmax(power[in_range])])


def local_log_residuals(distances, rolling_window):
    """Return log distances minus their rolling median.

    This keeps quiet and busy periods on a comparable scale.

    Args:
        distances: Array of consecutive distances.
        rolling_window: Width of the centered rolling median.

    Returns:
        Array with the same length as distances.
    """
    log_distances = np.log(distances)
    return log_distances - median_filter(log_distances, size=rolling_window, mode="nearest")


def boundary_scores(residuals):
    """Score each interior window by the residuals on both of its sides.

    Args:
        residuals: Array where residuals[t] describes the step from window t to t + 1.

    Returns:
        Array where element t scores window t + 1.
    """
    return residuals[:-1] + residuals[1:]


def otsu_threshold(values, bins=256):
    """Return the threshold that best separates values into two groups (Otsu's method).

    Args:
        values: Array of values.
        bins: Number of histogram bins.

    Returns:
        The bin center that maximizes the between-group variance.
    """
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = counts / counts.sum()
    lower_weight = np.cumsum(weights)
    lower_sum = np.cumsum(weights * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (lower_sum[-1] * lower_weight - lower_sum) ** 2 / (lower_weight * (1 - lower_weight))
    return float(centers[np.nanargmax(between)])


def detect_boundary_windows(scores, threshold):
    """Return one boundary window for each run of adjacent scores above a threshold.

    Args:
        scores: Output of boundary_scores.
        threshold: Scores above this value count as spikes.

    Returns:
        Sorted array of window indices. Each is the highest-scoring window in its run.
    """
    above = np.nonzero(scores > threshold)[0]
    if above.size == 0:
        return np.array([], dtype=np.int64)
    runs = np.split(above, np.nonzero(np.diff(above) > 1)[0] + 1)
    return np.array([run[np.argmax(scores[run])] + 1 for run in runs], dtype=np.int64)


def segment_length_counts(boundaries):
    """Return a dict of {segment length: count} for gaps between boundaries."""
    lengths, counts = np.unique(np.diff(boundaries), return_counts=True)
    return {int(length): int(count) for length, count in zip(lengths, counts)}


def drop_false_boundaries(boundaries, scores, max_length):
    """Remove weak boundaries that split what could be one day.

    While any two neighboring segments would fit in max_length windows together,
    this removes the lowest-scoring boundary between such a pair. The windows before
    the first boundary count as a segment too.

    Args:
        boundaries: Sorted array of boundary window indices.
        scores: Output of boundary_scores, where element t scores window t + 1.
        max_length: Longest allowed length for a merged segment.

    Returns:
        Sorted array of the boundaries that remain.
    """
    kept = np.array(boundaries, dtype=np.int64)
    while kept.size >= 2:
        previous = np.r_[-1, kept[:-2]]
        merged = kept[1:] - previous
        candidates = np.nonzero(merged <= max_length)[0]
        if candidates.size == 0:
            break
        weakest = candidates[np.argmin(scores[kept[candidates] - 1])]
        kept = np.delete(kept, weakest)
    return kept


def assign_day_slots(n_windows, boundaries, typical_length):
    """Assign a day and a slot to every window in the recovered order.

    Each day ends at a boundary window. Slot 0 is the boundary window, and slots
    count back along the recovered order from there. The last windows have no
    boundary after them, so they are placed as if their day had typical_length
    windows.

    Args:
        n_windows: Total number of windows.
        boundaries: Sorted array of boundary window indices.
        typical_length: Usual number of windows in a day.

    Returns:
        A tuple of (day_id, slot), integer arrays in recovered order.
    """
    boundaries = np.asarray(boundaries, dtype=np.int64)
    day_id = np.searchsorted(boundaries, np.arange(n_windows), side="left")
    slot = np.empty(n_windows, dtype=np.int64)
    has_boundary = day_id < boundaries.size
    slot[has_boundary] = boundaries[day_id[has_boundary]] - np.nonzero(has_boundary)[0]
    tail = np.nonzero(~has_boundary)[0]
    if tail.size:
        if tail.size > typical_length:
            raise ValueError(f"{tail.size} windows follow the last boundary, more than one day")
        slot[tail] = typical_length - 1 - (tail - tail[0])
    return day_id, slot

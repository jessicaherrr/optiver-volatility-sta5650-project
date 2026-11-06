"""Evaluation metrics for volatility forecasts."""

import numpy as np


def rmspe(y_true, y_pred):
    """Return the root mean squared percentage error, the competition metric.

    Formula: sqrt(mean(((y_true - y_pred) / y_true) ** 2)).

    Args:
        y_true: Array-like of actual values. All values must be nonzero.
        y_pred: Array-like of predictions, with the same shape as y_true.

    Returns:
        RMSPE as a float. A value of 0.1 means a typical error of about 10%.

    Raises:
        ValueError: If the shapes differ or y_true contains a zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes differ: {y_true.shape} and {y_pred.shape}")
    if np.any(y_true == 0):
        raise ValueError("y_true contains a zero, so RMSPE is undefined")
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

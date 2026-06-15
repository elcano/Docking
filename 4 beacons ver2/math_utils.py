"""Math utilities converted from MATLAB files: skewp.m, DCM.m, unit.m."""

from __future__ import annotations

import numpy as np


def as_col3(vector: np.ndarray | list | tuple) -> np.ndarray:
    """Return input as a 3-element float vector with shape (3,)."""
    arr = np.asarray(vector, dtype=float).reshape(-1) #convert to decimal umber
    if arr.size != 3:
        raise ValueError(f"Expected a 3-element vector, got shape {np.asarray(vector).shape}")
    return arr


def skewp(p: np.ndarray | list | tuple) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix for vector p."""
    p = as_col3(p)
    return np.array(
        [
            [0.0, -p[2], p[1]],
            [p[2], 0.0, -p[0]],
            [-p[1], p[0], 0.0],
        ],
        dtype=float,
    )


def dcm(p: np.ndarray | list | tuple) -> np.ndarray:
    """Direction cosine matrix converted from DCM.m."""
    p = as_col3(p)
    identity = np.eye(3)
    pcross = skewp(p)
    p_dot = float(p @ p) # dot product, matrix multiply
    numerator = 8.0 * (pcross @ pcross) - 4.0 * (1.0 - p_dot) * pcross
    denominator = (1.0 + p_dot) ** 2
    return identity + numerator / denominator


def unit(bmeasure: np.ndarray | list | tuple) -> np.ndarray:
    """Normalize each row of a measurement matrix to unit length."""
    bmeasure = np.asarray(bmeasure, dtype=float)
    if bmeasure.ndim != 2:
        raise ValueError("bmeasure must be a 2D array")

    # Computes the length of each row.
    norms = np.linalg.norm(bmeasure, axis=1, keepdims=True)
    
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a row with zero norm")
    return bmeasure / norms

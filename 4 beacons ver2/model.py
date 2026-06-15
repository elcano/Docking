"""Core beacon solver math converted from Bibguess.m, pos_partial.m, orient_partial.m, MSM.m."""

from __future__ import annotations

import numpy as np

from math_utils import as_col3, dcm, skewp


def bibguess(p: np.ndarray | list | tuple, rs: np.ndarray | list | tuple, ri: np.ndarray | list | tuple) -> tuple[np.ndarray, np.ndarray]:
    """Compute Bi and bguess for one beacon."""
    rs = as_col3(rs) # target/camera position guess
    ri = as_col3(ri) # known beacon position

    diff = ri - rs
    denom = float(diff @ diff)
    if denom == 0:
        raise ValueError("Ri and Rs are identical; cannot compute line-of-sight unit vector")

    bi = diff / np.sqrt(denom)
    bguess = dcm(p) @ bi
    return bi, bguess


def pos_partial(bi: np.ndarray | list | tuple, c: np.ndarray, rs: np.ndarray | list | tuple, ri: np.ndarray | list | tuple) -> np.ndarray:
    """Position partial derivative matrix converted from pos_partial.m."""
    bi = as_col3(bi)
    rs = as_col3(rs)
    ri = as_col3(ri)

    denom = np.linalg.norm(ri - rs)
    if denom == 0:
        raise ValueError("Ri and Rs are identical; position partial is undefined")

    identity = np.eye(3)
    return (-(c @ (identity - np.outer(bi, bi)))) / denom


def orient_partial(bguess: np.ndarray | list | tuple, p: np.ndarray | list | tuple) -> np.ndarray:
    """Orientation partial derivative matrix converted from orient_partial.m."""
    bguess = as_col3(bguess)
    p = as_col3(p)

    identity = np.eye(3)
    bcross = skewp(bguess)
    pcross = skewp(p)
    p_dot = float(p @ p)

    leftside = (4.0 / (1.0 + p_dot) ** 2) * bcross
    rightside = (1.0 - p_dot) * identity - 2.0 * pcross + 2.0 * np.outer(p, p)
    return leftside @ rightside


def msm(p: np.ndarray | list | tuple, rs: np.ndarray | list | tuple, bmeasure: np.ndarray | list | tuple, ri: np.ndarray | list | tuple, n_beacons: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build measurement sensitivity matrix H and residual vector deltab."""
    p = as_col3(p)
    rs = as_col3(rs)
    ri = np.asarray(ri, dtype=float)
    bmeasure = np.asarray(bmeasure, dtype=float).reshape(-1)

    if ri.shape[0] < n_beacons or ri.shape[1] != 3:
        raise ValueError(f"ri must have at least {n_beacons} rows and exactly 3 columns")
    expected_measurements = 3 * n_beacons
    if bmeasure.size != expected_measurements:
        raise ValueError(f"bmeasure must contain {expected_measurements} values, got {bmeasure.size}")

    c = dcm(p)
    h_blocks: list[np.ndarray] = []
    bguess_all: list[np.ndarray] = []

    for i in range(n_beacons):
        beacon_ri = ri[i, :]
        bi, bguess = bibguess(p, rs, beacon_ri)
        hi_rs = pos_partial(bi, c, rs, beacon_ri)
        hi_p = orient_partial(bguess, p)
        h_blocks.append(np.hstack((hi_rs, hi_p)))
        bguess_all.append(bguess)

    h = np.vstack(h_blocks)
    bguess_vector = np.concatenate(bguess_all) # need to check
    deltab = bmeasure - bguess_vector
    return h, deltab

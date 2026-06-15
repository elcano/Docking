"""Runnable four-beacon solver."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from io_utils import SerialNoDeviceError, get_config, get_serial
from math_utils import dcm, unit
from model import msm


@dataclass
class SolveResult:
    rs: np.ndarray
    p: np.ndarray
    iterations: int
    converged: bool


def prepare_bmeasure_vector(bmeasure: np.ndarray, focal_length: float = 320.0) -> np.ndarray:
    """Apply focal length and flatten measurements beacon by beacon."""
    bmeasure = np.asarray(bmeasure, dtype=float).copy()
    if bmeasure.ndim != 2 or bmeasure.shape[1] != 3:
        raise ValueError("bmeasure must be a matrix with 3 columns")

    if np.all(bmeasure[:, 0] == 0):
        bmeasure[:, 0] = focal_length

    b_unit = unit(bmeasure)
    return b_unit.reshape(-1)


def has_real_measurements(bmeasure: np.ndarray) -> bool:
    """Return True when bmeasure has non-zero camera/image-plane data."""
    bmeasure = np.asarray(bmeasure, dtype=float)
    return bmeasure.ndim == 2 and bmeasure.shape[1] == 3 and np.any(bmeasure[:, 1:] != 0)


def print_pose_details(result: SolveResult) -> None:
    """Print pose plus derived values that are easier to compare to a tape measure."""
    distance_m = float(np.linalg.norm(result.rs))
    object_center_camera_frame = dcm(result.p) @ (-result.rs)

    print("object center:", object_center_camera_frame)
    print("Rs:", result.rs)
    print("p:", result.p)
    print(f"distance: {distance_m:.4f} m ({distance_m * 3.28084:.2f} ft)")
    # print("object center in camera frame:", object_center_camera_frame)


def solve_pose(
    bmeasure: np.ndarray,
    ri: np.ndarray,
    guess_rs: np.ndarray | None = None,
    guess_p: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    tolerance: float = 1e-5,
    max_iters: int = 200,
    focal_length: float = 320.0,
) -> SolveResult:
    """Run the iterative least-squares pose solve."""
    guess_rs = np.zeros(3, dtype=float) if guess_rs is None else np.asarray(guess_rs, dtype=float).reshape(3)
    guess_p = np.zeros(3, dtype=float) if guess_p is None else np.asarray(guess_p, dtype=float).reshape(3)
    ri = np.asarray(ri, dtype=float)
    n_beacons = ri.shape[0]

    b_unit_vector = prepare_bmeasure_vector(bmeasure, focal_length=focal_length)
    weight = np.eye(b_unit_vector.size) if weight is None else np.asarray(weight, dtype=float)

    for iteration in range(1, max_iters + 1):
        if not np.all(np.isfinite(guess_rs)) or not np.all(np.isfinite(guess_p)):
            return SolveResult(rs=guess_rs, p=guess_p, iterations=iteration, converged=False)

        if np.linalg.norm(guess_rs) > 1e6 or np.linalg.norm(guess_p) > 1e6:
            return SolveResult(rs=guess_rs, p=guess_p, iterations=iteration, converged=False)

        try:
            h, deltab = msm(guess_p, guess_rs, b_unit_vector, ri, n_beacons=n_beacons)
        except OverflowError:
            return SolveResult(rs=guess_rs, p=guess_p, iterations=iteration, converged=False)

        normal_matrix = h.T @ weight @ h
        rhs = h.T @ weight @ deltab

        try:
            deltax = np.linalg.solve(normal_matrix, rhs)
        except np.linalg.LinAlgError:
            deltax = np.linalg.lstsq(normal_matrix, rhs, rcond=None)[0]

        if not np.all(np.isfinite(deltax)):
            return SolveResult(rs=guess_rs, p=guess_p, iterations=iteration, converged=False)

        guess_rs = guess_rs + deltax[:3]
        guess_p = guess_p + deltax[3:6]

        if np.linalg.norm(deltax) < tolerance:
            return SolveResult(rs=guess_rs, p=guess_p, iterations=iteration, converged=True)

    return SolveResult(rs=guess_rs, p=guess_p, iterations=max_iters, converged=False)


def run_loop(
    csv_path: str | None = None,
    use_serial: bool = True,
    focal_length: float = 320.0,
    tolerance: float = 1e-5,
    max_iters: int = 200,
    once: bool = False,
    port: str | None = None,
    baud_rate: int = 115200,
    replay_file: str | None = None,
    min_beacons: int | None = None,
    serial_timeout: float | None = None,
    serial_input: str = "unlabeled",
    serial_format: str = "raw",
    unlabeled_method: str = "quadrant",
    cluster_samples: int = 300,
    corner_fraction: float = 0.25,
    quadrant_order: str = "TR,BR,BL,TL",
    image_width: float = 320.0,
    image_height: float = 240.0,
    flip_y: bool = True,
    stable_samples: int = 3,
    stable_radius: float = 0.03,
    pivot_repeat_radius: float = 0.25,
    pivot_min_distance: float = 0.5,
    pivot_neighbor: str = "auto-x",
) -> None:
    """Run the MASTERLOOP-style workflow for four beacons."""
    bmeasure, ri = get_config(csv_path)

    if ri.shape[0] != 4:
        raise ValueError(f"Four-beacon solver requires exactly 4 CSV rows, got {ri.shape[0]}")

    if min_beacons is None:
        min_beacons = ri.shape[0]

    while True:
        if use_serial:
            try:
                bmeasure = get_serial(
                    port=port,
                    baud_rate=baud_rate,
                    replay_file=replay_file,
                    rows=ri.shape[0],
                    min_required=min_beacons,
                    timeout_seconds=serial_timeout,
                    serial_input=serial_input,
                    serial_format=serial_format,
                    unlabeled_method=unlabeled_method,
                    cluster_samples=cluster_samples,
                    corner_fraction=corner_fraction,
                    quadrant_order=quadrant_order,
                    image_width=image_width,
                    image_height=image_height,
                    flip_y=flip_y,
                    stable_samples=stable_samples,
                    stable_radius=stable_radius,
                    pivot_repeat_radius=pivot_repeat_radius,
                    pivot_min_distance=pivot_min_distance,
                    pivot_neighbor=pivot_neighbor,
                )
            except SerialNoDeviceError as exc:
                print(exc)
                if once:
                    break
                print("Waiting for a serial device...")
                time.sleep(1)
                continue
            except Exception as exc:
                print(f"Serial read failed with error: {exc}")
                if once:
                    break
                print("Waiting for better serial data...")
                time.sleep(1)
                continue
        else:
            if not has_real_measurements(bmeasure):
                print("No serial input is being used, and the CSV does not contain measurement data.")
                print("This is only a config file, so the solver will not run.")
                break

        result = solve_pose(
            bmeasure=bmeasure,
            ri=ri,
            tolerance=tolerance,
            max_iters=max_iters,
            focal_length=focal_length,
        )

        if result.converged:
            print_pose_details(result)
        else:
            print("Solver did not converge with the current measurements.")
            print("Last Rs:", result.rs)
            print("Last p:", result.p)

        if once:
            break

"""I/O helpers for the four-beacon solver.

CSV data and serial data have different meanings:
- CSV columns 2-4 are optional measurement values from a file.
- CSV last three columns are real beacon positions Ri = [depth, Y, Z] in meters.
- Indexed serial lines are OpenMV measurements: beacon_number,y,z.
- Unlabeled serial lines are OpenMV measurements: y,z. In this mode the code
  filters stable point clusters and infers beacon rows from this pivot sequence:
  1,2,3,4,1,4,3,2.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from position_data import add_point
except ImportError:
    add_point = None


class SerialNoDeviceError(RuntimeError):
    """Raised when no serial device is available."""


class ReplayInput:
    """Serial-like reader for replaying local text files."""

    def __init__(self, replay_file: str | Path) -> None:
        self.replay_file = Path(replay_file)
        self.file = None

    def __enter__(self) -> "ReplayInput":
        self.file = self.replay_file.open("rb")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.file is not None:
            self.file.close()

    def readline(self) -> bytes:
        if self.file is None:
            return b""

        line = self.file.readline()
        if line:
            return line

        self.file.seek(0)
        return self.file.readline()


def get_config(csv_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a four-beacon configuration CSV."""
    if csv_path is None:
        try:
            from tkinter import Tk, filedialog
        except Exception as exc:
            raise ValueError("csv_path is required when tkinter is unavailable") from exc

        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="Select the 4-beacon configuration CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()

        if not selected:
            raise RuntimeError("Program Terminated: No configuration file was selected.")

        csv_path = selected

    csv_path = Path(csv_path)
    config_data = pd.read_csv(csv_path)

    if config_data.shape[1] < 7:
        raise ValueError("Configuration CSV must have at least 7 columns")

    if config_data.shape[0] != 4:
        raise ValueError(
            f"Four-beacon configuration must have exactly 4 rows, got {config_data.shape[0]}"
        )

    bmeasure = config_data.iloc[:, 1:4].to_numpy(dtype=float)
    ri = config_data.iloc[:, -3:].to_numpy(dtype=float)

    print(f"Successfully loaded configuration from: {csv_path.name}")
    print("Loaded 4 beacon positions from last three CSV columns.")

    return bmeasure, ri


def list_serial_ports() -> list[str]:
    """Return available serial port names."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise ImportError(
            "pyserial is required for serial input. Install it with: pip install pyserial"
        ) from exc

    return [p.device for p in list_ports.comports()]


def parse_sensor_line(raw_data: str, rows: int) -> tuple[int, float, float] | None:
    """Parse one indexed sensor line: beacon_number,y,z."""
    raw_data = raw_data.strip()

    if not raw_data:
        return None

    parts = raw_data.split(",")

    if len(parts) < 3:
        print(f"Serial warning: ignored invalid line: {raw_data!r}")
        return None

    try:
        beacon_number = int(round(float(parts[0])))
        y_val = float(parts[1])
        z_val = float(parts[2])
    except ValueError:
        print(f"Serial warning: ignored non-numeric line: {raw_data!r}")
        return None

    if not 1 <= beacon_number <= rows:
        print(f"Serial warning: invalid beacon number {beacon_number}. Line ignored.")
        return None

    return beacon_number - 1, y_val, z_val


def parse_unlabeled_sensor_line(raw_data: str) -> tuple[float, float] | None:
    """Parse one unlabeled sensor line: y,z."""
    raw_data = raw_data.strip()

    if not raw_data:
        return None

    parts = raw_data.split(",")

    if len(parts) != 2:
        print(f"Serial warning: ignored invalid unlabeled line: {raw_data!r}")
        return None

    try:
        y_val = float(parts[0])
        z_val = float(parts[1])
    except ValueError:
        print(f"Serial warning: ignored non-numeric unlabeled line: {raw_data!r}")
        return None

    return y_val, z_val


def parse_quadrant_order(quadrant_order: str) -> list[str]:
    """Return quadrant labels for beacon rows 1..4."""
    labels = [
        part.strip().upper()
        for part in quadrant_order.replace(";", ",").split(",")
        if part.strip()
    ]

    valid = {"TR", "BR", "BL", "TL"}

    if len(labels) != 4 or set(labels) != valid:
        raise ValueError(
            'quadrant_order must contain TR,BR,BL,TL exactly once, for example "TR,BR,BL,TL"'
        )

    return labels


def assign_quadrant_beacons(
    points: list[tuple[float, float]],
    corner_fraction: float = 0.25,
    quadrant_order: str = "TR,BR,BL,TL",
    min_points_per_quadrant: int = 5,
) -> list[tuple[int, float, float]]:
    """Estimate four beacon corners from a batch of unlabeled PSD points."""
    if not 0.0 < corner_fraction <= 1.0:
        raise ValueError("corner_fraction must be greater than 0 and at most 1")

    arr = np.asarray(points, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be a list of y,z pairs")

    if arr.shape[0] < 4 * min_points_per_quadrant:
        raise ValueError("Not enough points for quadrant assignment")

    x_values = arr[:, 0]
    y_values = arr[:, 1]

    low_x, high_x = np.percentile(x_values, [5, 95])
    low_y, high_y = np.percentile(y_values, [5, 95])

    mid_x = (low_x + high_x) / 2.0
    mid_y = (low_y + high_y) / 2.0

    masks = {
        "TR": (x_values >= mid_x) & (y_values >= mid_y),
        "BR": (x_values >= mid_x) & (y_values < mid_y),
        "BL": (x_values < mid_x) & (y_values < mid_y),
        "TL": (x_values < mid_x) & (y_values >= mid_y),
    }

    ideal_corners = {
        "TR": np.array([high_x, high_y], dtype=float),
        "BR": np.array([high_x, low_y], dtype=float),
        "BL": np.array([low_x, low_y], dtype=float),
        "TL": np.array([low_x, high_y], dtype=float),
    }

    centers: dict[str, np.ndarray] = {}

    for label, mask in masks.items():
        quadrant_points = arr[mask]

        if quadrant_points.shape[0] < min_points_per_quadrant:
            raise ValueError(
                f"Not enough points in {label} quadrant. "
                f"Got {quadrant_points.shape[0]}, need at least {min_points_per_quadrant}."
            )

        distances = np.linalg.norm(quadrant_points - ideal_corners[label], axis=1)

        keep_count = max(
            min_points_per_quadrant,
            int(np.ceil(quadrant_points.shape[0] * corner_fraction)),
        )
        keep_count = min(keep_count, quadrant_points.shape[0])

        closest = quadrant_points[np.argsort(distances)[:keep_count]]
        centers[label] = np.median(closest, axis=0)

    ordered_labels = parse_quadrant_order(quadrant_order)

    print(
        "Quadrant centers: "
        + ", ".join(
            f"{label}=({centers[label][0]:.4f}, {centers[label][1]:.4f})"
            for label in ["TR", "BR", "BL", "TL"]
        )
    )

    print(
        "Quadrant assignment: "
        + ", ".join(
            f"Beacon {i + 1}={label}"
            for i, label in enumerate(ordered_labels)
        )
    )

    return [
        (row_index, centers[label][0], centers[label][1])
        for row_index, label in enumerate(ordered_labels)
    ]


class UnlabeledBeaconTracker:
    """Convert a fast unlabeled y,z stream into four labeled beacon rows."""

    sequence = [0, 1, 2, 3, 0, 3, 2, 1]

    def __init__(
        self,
        rows: int,
        stable_samples: int = 3,
        stable_radius: float = 0.03,
        pivot_repeat_radius: float = 0.25,
        pivot_min_distance: float = 0.5,
        pivot_neighbor: str = "auto-x",
    ) -> None:
        if rows != 4:
            raise ValueError("Four-beacon unlabeled tracking requires exactly 4 beacons")

        if stable_samples < 2:
            raise ValueError("stable_samples must be at least 2")

        if pivot_neighbor not in {"auto-x", "auto-x-inverted", "beacon2", "beacon4"}:
            raise ValueError(
                'pivot_neighbor must be "auto-x", "auto-x-inverted", "beacon2", or "beacon4"'
            )

        self.stable_samples = stable_samples
        self.stable_radius = stable_radius
        self.pivot_repeat_radius = pivot_repeat_radius
        self.pivot_min_distance = pivot_min_distance
        self.pivot_neighbor = pivot_neighbor

        self.window: list[np.ndarray] = []
        self.recent_stable: list[np.ndarray] = []

        self.synced = False
        self.sequence_index = 0

        self.release_radius = max(stable_radius * 3.0, 0.1)
        self.last_stable_point: np.ndarray | None = None
        self.waiting_for_movement = False

    def add_point(self, y_val: float, z_val: float) -> list[tuple[int, float, float]]:
        stable_point = self._stable_point(y_val, z_val)

        if stable_point is None:
            return []

        if self.synced:
            row_index = self.sequence[self.sequence_index]
            self.sequence_index = (self.sequence_index + 1) % len(self.sequence)

            return [(row_index, stable_point[0], stable_point[1])]

        return self._sync_from_pivot(stable_point)

    def _stable_point(self, y_val: float, z_val: float) -> np.ndarray | None:
        point = np.array([y_val, z_val], dtype=float)

        if self.waiting_for_movement and self.last_stable_point is not None:
            if np.linalg.norm(point - self.last_stable_point) <= self.release_radius:
                self.window.clear()
                return None

            self.waiting_for_movement = False
            self.window.clear()

        self.window.append(point)

        if len(self.window) > self.stable_samples:
            self.window.pop(0)

        if len(self.window) < self.stable_samples:
            return None

        points = np.vstack(self.window)
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)

        if float(np.max(distances)) > self.stable_radius:
            return None

        self.window.clear()
        self.last_stable_point = center
        self.waiting_for_movement = True

        return center

    def _sync_from_pivot(self, stable_point: np.ndarray) -> list[tuple[int, float, float]]:
        self.recent_stable.append(stable_point)

        if len(self.recent_stable) > 3:
            self.recent_stable.pop(0)

        if len(self.recent_stable) < 3:
            print(
                f"Stable point found while syncing: "
                f"({stable_point[0]:.4f}, {stable_point[1]:.4f})"
            )
            return []

        repeated_before, pivot, repeated_after = self.recent_stable

        repeat_distance = np.linalg.norm(repeated_before - repeated_after)

        pivot_distance = min(
            np.linalg.norm(repeated_before - pivot),
            np.linalg.norm(repeated_after - pivot),
        )

        print(
            "Stable point found while syncing: "
            f"({stable_point[0]:.4f}, {stable_point[1]:.4f}); "
            f"pivot check repeat={repeat_distance:.4f}, pivot_dist={pivot_distance:.4f}"
        )

        if repeat_distance > self.pivot_repeat_radius or pivot_distance < self.pivot_min_distance:
            return []

        neighbor_index = self._infer_repeated_neighbor(pivot, repeated_after)

        if neighbor_index == 1:
            self.sequence_index = 2
        else:
            self.sequence_index = 6

        self.synced = True
        self.recent_stable.clear()

        print(f"Pivot sync found: beacon 1 plus repeated beacon {neighbor_index + 1}.")

        return [
            (0, pivot[0], pivot[1]),
            (neighbor_index, repeated_after[0], repeated_after[1]),
        ]

    def _infer_repeated_neighbor(self, pivot: np.ndarray, repeated: np.ndarray) -> int:
        if self.pivot_neighbor == "beacon2":
            return 1

        if self.pivot_neighbor == "beacon4":
            return 3

        if self.pivot_neighbor == "auto-x-inverted":
            return 3 if repeated[0] >= pivot[0] else 1

        return 1 if repeated[0] >= pivot[0] else 3


def get_serial(
    port: str | None = None,
    baud_rate: int = 115200,
    replay_file: str | Path | None = None,
    rows: int = 4,
    min_required: int | None = None,
    timeout_seconds: float | None = None,
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
) -> np.ndarray:
    """Read serial measurements until enough unique beacon rows are received."""
    if rows != 4:
        raise ValueError("Four-beacon serial input requires rows=4")

    if min_required is None:
        min_required = rows

    if min_required < 1 or min_required > rows:
        raise ValueError("min_required must be between 1 and rows")

    if serial_format not in {"raw", "pixel"}:
        raise ValueError('serial_format must be either "raw" or "pixel"')

    if serial_input not in {"indexed", "unlabeled"}:
        raise ValueError('serial_input must be either "indexed" or "unlabeled"')

    if unlabeled_method not in {"quadrant", "pivot"}:
        raise ValueError('unlabeled_method must be either "quadrant" or "pivot"')

    if cluster_samples < 4:
        raise ValueError("cluster_samples must be at least 4")

    parse_quadrant_order(quadrant_order)

    if replay_file is None:
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "pyserial is required for serial input. Install it with: pip install pyserial"
            ) from exc
    else:
        serial = None
        replay_path = Path(replay_file)

        if not replay_path.exists():
            raise FileNotFoundError(f"Replay file not found: {replay_path}")

    if replay_file is None and port is None:
        available_ports = list_serial_ports()

        if not available_ports:
            raise SerialNoDeviceError("No serial connections detected. Looking for file instead.")

        port = available_ports[0]
        print(f"Available serial ports: {available_ports}")

    if replay_file is None:
        print(f"Connecting to device on {port} at {baud_rate} baud...")
    else:
        print(f"Replaying serial data from: {Path(replay_file)}")

    print(f"Waiting for {min_required} unique beacon measurement(s)...")

    if serial_input == "indexed":
        print('Expected format: "beacon_number,y,z", for example: "1,36,32"')
    else:
        print('Expected format: "y,z", for example: "0.0000,-6.5000"')
        print(f"Unlabeled method: {unlabeled_method}")

        if unlabeled_method == "quadrant":
            print(f"Quadrant mode: collecting {cluster_samples} samples before assignment.")
        else:
            print("Pivot sequence: 1,2,3,4,1,4,3,2")
            print(
                "Unlabeled filter: "
                f"{stable_samples} samples within {stable_radius} radius; "
                f"pivot repeat radius {pivot_repeat_radius}"
            )

    print(f"Serial format: {serial_format}")

    bmeasure = np.zeros((rows, 3), dtype=float)
    received = np.zeros(rows, dtype=bool)

    start_time = time.time()

    tracker = (
        UnlabeledBeaconTracker(
            rows=rows,
            stable_samples=stable_samples,
            stable_radius=stable_radius,
            pivot_repeat_radius=pivot_repeat_radius,
            pivot_min_distance=pivot_min_distance,
            pivot_neighbor=pivot_neighbor,
        )
        if serial_input == "unlabeled" and unlabeled_method == "pivot"
        else None
    )

    input_source = (
        ReplayInput(replay_file)
        if replay_file is not None
        else serial.Serial(port, baud_rate, timeout=1)
    )

    with input_source as open_mv:
        if serial_input == "unlabeled" and unlabeled_method == "quadrant":
            raw_points: list[tuple[float, float]] = []

            while len(raw_points) < cluster_samples:
                if timeout_seconds is not None and time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting for serial data. "
                        f"Collected {len(raw_points)} of {cluster_samples} samples."
                    )

                raw_data = open_mv.readline().decode("utf-8", errors="replace").strip()

                parsed_unlabeled = parse_unlabeled_sensor_line(raw_data)

                if parsed_unlabeled is None:
                    continue

                y_val, z_val = parsed_unlabeled

                if add_point is not None:
                    add_point(y_val, z_val)

                raw_points.append(parsed_unlabeled)

            accepted = assign_quadrant_beacons(
                raw_points,
                corner_fraction=corner_fraction,
                quadrant_order=quadrant_order,
            )

            for row_index, y_val, z_val in accepted:
                if serial_format == "pixel":
                    pixel_x = y_val
                    pixel_y = z_val
                    y_measure = pixel_x - image_width / 2.0

                    if flip_y:
                        z_measure = image_height / 2.0 - pixel_y
                    else:
                        z_measure = pixel_y - image_height / 2.0
                else:
                    y_measure = y_val
                    z_measure = z_val

                bmeasure[row_index, 1] = y_measure
                bmeasure[row_index, 2] = z_measure
                received[row_index] = True

                print(
                    f"Beacon {row_index + 1}: raw=({y_val}, {z_val}), "
                    f"measurement=({y_measure}, {z_measure}) "
                    f"[{np.count_nonzero(received)}/{min_required}]"
                )

            print("Matrix bmeasure successfully populated.")

            return bmeasure

        while np.count_nonzero(received) < min_required:
            if timeout_seconds is not None and time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for serial data. "
                    f"Received {np.count_nonzero(received)} of {min_required} required beacons."
                )

            raw_data = open_mv.readline().decode("utf-8", errors="replace").strip()

            if serial_input == "indexed":
                parsed = parse_sensor_line(raw_data, rows)

                if parsed is None:
                    continue

                row_index, y_val, z_val = parsed
                accepted = [(row_index, y_val, z_val)]

            else:
                parsed_unlabeled = parse_unlabeled_sensor_line(raw_data)

                if parsed_unlabeled is None:
                    continue

                y_val, z_val = parsed_unlabeled

                if add_point is not None:
                    add_point(y_val, z_val)

                assert tracker is not None
                accepted = tracker.add_point(y_val, z_val)

            for row_index, y_val, z_val in accepted:
                if serial_format == "pixel":
                    pixel_x = y_val
                    pixel_y = z_val
                    y_measure = pixel_x - image_width / 2.0

                    if flip_y:
                        z_measure = image_height / 2.0 - pixel_y
                    else:
                        z_measure = pixel_y - image_height / 2.0
                else:
                    y_measure = y_val
                    z_measure = z_val

                bmeasure[row_index, 1] = y_measure
                bmeasure[row_index, 2] = z_measure
                received[row_index] = True

                print(
                    f"Beacon {row_index + 1}: raw=({y_val}, {z_val}), "
                    f"measurement=({y_measure}, {z_measure}) "
                    f"[{np.count_nonzero(received)}/{min_required}]"
                )

    print("Matrix bmeasure successfully populated.")

    return bmeasure
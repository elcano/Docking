"""Command-line entry point for the four-beacon solver."""

from __future__ import annotations

import argparse
import threading

from solver import run_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-beacon solver")

    parser.add_argument("--config", help="Path to 4-beacon configuration CSV.")
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--port", help="Serial port, for example COM4.")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--replay-file")
    parser.add_argument("--min-beacons", type=int)
    parser.add_argument("--serial-timeout", type=float)

    parser.add_argument(
        "--serial-input",
        choices=["indexed", "unlabeled"],
        default="unlabeled",
    )

    parser.add_argument(
        "--serial-format",
        choices=["raw", "pixel"],
        default="raw",
    )

    parser.add_argument(
        "--unlabeled-method",
        choices=["quadrant", "pivot"],
        default="quadrant",
    )

    parser.add_argument("--cluster-samples", type=int, default=60)
    parser.add_argument("--corner-fraction", type=float, default=0.25)
    parser.add_argument("--quadrant-order", default="TR,BR,BL,TL")

    parser.add_argument("--image-width", type=float, default=320.0)
    parser.add_argument("--image-height", type=float, default=240.0)
    parser.add_argument("--no-flip-y", action="store_true")

    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--stable-radius", type=float, default=0.03)
    parser.add_argument("--pivot-repeat-radius", type=float, default=0.25)
    parser.add_argument("--pivot-min-distance", type=float, default=0.5)

    parser.add_argument(
        "--pivot-neighbor",
        choices=["auto-x", "auto-x-inverted", "beacon2", "beacon4"],
        default="auto-x",
    )

    parser.add_argument("--focal-length", type=float, default=26.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--once", action="store_true")

    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Show live Y,Z plot while solver is running.",
    )

    return parser.parse_args()


def run_solver(args):
    run_loop(
        csv_path=args.config,
        use_serial=not args.no_serial,
        focal_length=args.focal_length,
        tolerance=args.tolerance,
        max_iters=args.max_iters,
        once=args.once,
        port=args.port,
        baud_rate=args.baud_rate,
        replay_file=args.replay_file,
        min_beacons=args.min_beacons,
        serial_timeout=args.serial_timeout,
        serial_input=args.serial_input,
        serial_format=args.serial_format,
        unlabeled_method=args.unlabeled_method,
        cluster_samples=args.cluster_samples,
        corner_fraction=args.corner_fraction,
        quadrant_order=args.quadrant_order,
        image_width=args.image_width,
        image_height=args.image_height,
        flip_y=not args.no_flip_y,
        stable_samples=args.stable_samples,
        stable_radius=args.stable_radius,
        pivot_repeat_radius=args.pivot_repeat_radius,
        pivot_min_distance=args.pivot_min_distance,
        pivot_neighbor=args.pivot_neighbor,
    )


def main() -> None:
    args = parse_args()

    if args.live_plot:
        from position_data import start_plot

        solver_thread = threading.Thread(
            target=run_solver,
            args=(args,),
            daemon=True
        )
        solver_thread.start()

        start_plot()
    else:
        run_solver(args)


if __name__ == "__main__":
    main()
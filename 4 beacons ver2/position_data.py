from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TRAIL_LENGTH = 100
points = deque(maxlen=TRAIL_LENGTH)

fig = None
ax = None
scatter = None
line_plot = None
colors = None


def add_point(x, y):
    points.append((x, y))


def start_plot():
    global fig, ax, scatter, line_plot, colors

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.set_xticks(np.arange(-5, 5.1, 0.1), minor=True)
    ax.set_yticks(np.arange(-5, 5.1, 0.1), minor=True)

    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.6)

    ax.axhline(0, color="black", lw=1.0)
    ax.axvline(0, color="black", lw=1.0)

    ax.set_title("Live Serial Position Tracker")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    colors = np.zeros((TRAIL_LENGTH, 4))
    for i in range(TRAIL_LENGTH):
        t = i / (TRAIL_LENGTH - 1)
        colors[i] = (1 - t, 0, t, 1)

    scatter = ax.scatter([], [], s=3)
    line_plot, = ax.plot([], [], lw=0.8, color="cyan", alpha=0.6)

    ani = FuncAnimation(
        fig,
        update_plot,
        interval=20,
        blit=True,
        cache_frame_data=False
    )

    plt.show()


def update_plot(_):
    if len(points) > 0:
        data = np.array(points)

        scatter.set_offsets(data)
        scatter.set_color(colors[-len(data):])
        line_plot.set_data(data[:, 0], data[:, 1])

    return scatter, line_plot
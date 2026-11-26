"""Visualization helpers for the double-pendulum simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
import numpy as np


def animate_double_pendulum(
    t_plot: np.ndarray,
    x_plot: np.ndarray,
    *,
    l1: float = 1.0,
    l2: float = 1.0,
    cart_width: float = 0.6,
    cart_height: float = 0.3,
    fps: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, FuncAnimation]:
    """Animate a simulated cart with a double pendulum.

    Parameters
    ----------
    t_plot:
        One-dimensional time array.
    x_plot:
        State array with columns ``[z, theta1, theta2, zd, theta1d,
        theta2d]``. The initial state may be included as an extra first row,
        as it is in :mod:`double_pendulum.sim`.
    l1, l2:
        Lengths of the two pendulum rods.
    cart_width, cart_height:
        Dimensions of the cart drawn beneath the pivot.
    fps:
        Output/display frame rate. If omitted, it is estimated from
        ``t_plot``.
    save_path:
        Optional GIF or MP4 output path. GIF output requires Pillow; MP4
        output requires an available FFmpeg installation.
    show:
        Call ``plt.show()`` before returning when true.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]
        The figure and animation, allowing callers to further customize or
        save them.
    """
    t = np.asarray(t_plot, dtype=float).reshape(-1)
    states = np.asarray(x_plot, dtype=float)

    if t.ndim != 1 or t.size < 1:
        raise ValueError("t_plot must contain at least one time value")
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("x_plot must be a 2D array with at least 3 columns")

    # sim.py stores the initial state before appending one state per time step.
    if states.shape[0] == t.size + 1:
        states = states[:-1]
    elif states.shape[0] != t.size:
        raise ValueError("x_plot must have len(t_plot) or len(t_plot) + 1 rows")

    if l1 <= 0 or l2 <= 0:
        raise ValueError("l1 and l2 must be positive")

    z = states[:, 0]
    theta1 = states[:, 1]
    theta2 = states[:, 2]

    # Angles are measured from the downward vertical.
    x1 = z + l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    horizontal_extent = max(l1 + l2, float(np.ptp(z)) + cart_width)
    margin = 0.15 * max(horizontal_extent, l1 + l2)
    ax.set_xlim(float(np.min(np.minimum(z - cart_width / 2, x2))) - margin,
                float(np.max(np.maximum(z + cart_width / 2, x2))) + margin)
    # Leave room for the rods to point straight up as well as straight down.
    ax.set_ylim(-(l1 + l2) - margin, (l1 + l2) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.axis("off")

    line_color = "black"
    rod1, = ax.plot([], [], color=line_color, lw=1.8, solid_capstyle="round",
                    zorder=6)
    rod2, = ax.plot([], [], color=line_color, lw=1.8, solid_capstyle="round",
                    zorder=6)
    rail, = ax.plot([], [], color=line_color, lw=1.2, zorder=1)
    pivot = Circle((0, 0), radius=0.055, facecolor="white", edgecolor=line_color,
                   lw=1.4, zorder=4)
    bob1 = Circle((0, 0), radius=0.105, facecolor="white", edgecolor=line_color,
                  lw=1.6, zorder=4)
    bob2 = Circle((0, 0), radius=0.105, facecolor="white", edgecolor=line_color,
                  lw=1.6, zorder=4)
    cart = Rectangle((0, 0), cart_width, cart_height, facecolor="white",
                     edgecolor=line_color, lw=1.5, zorder=3)
    wheel_radius = min(0.06, cart_height * 0.18)
    wheel1 = Circle((0, 0), radius=wheel_radius, facecolor=line_color,
                    edgecolor=line_color, lw=1.2, zorder=4)
    wheel2 = Circle((0, 0), radius=wheel_radius, facecolor=line_color,
                    edgecolor=line_color, lw=1.2, zorder=4)
    ground_y = -cart_height - 2 * wheel_radius
    ground, = ax.plot([], [], color=line_color, lw=1.2, zorder=1)
    ground_hatching = []
    ground_x_min, ground_x_max = ax.get_xlim()
    hatch_spacing = 0.12
    hatch_depth = 0.08
    for hatch_x in np.arange(ground_x_min, ground_x_max, hatch_spacing):
        hatch, = ax.plot(
            [hatch_x, hatch_x + hatch_depth],
            [ground_y, ground_y - hatch_depth],
            color=line_color,
            lw=0.8,
            zorder=1,
        )
        ground_hatching.append(hatch)
    ax.add_patch(pivot)
    ax.add_patch(bob1)
    ax.add_patch(bob2)
    ax.add_patch(cart)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)

    bob1_label = ax.text(0, 0, r"$m_1$", ha="center", va="center", fontsize=10,
                         zorder=8, math_fontfamily="cm")
    bob2_label = ax.text(0, 0, r"$m_2$", ha="center", va="center", fontsize=10,
                         zorder=8, math_fontfamily="cm")
    rod1_label = ax.text(0, 0, r"$\ell_1$", ha="center", va="center",
                         fontsize=11, zorder=7, math_fontfamily="cm")
    rod2_label = ax.text(0, 0, r"$\ell_2$", ha="center", va="center",
                         fontsize=11, zorder=7, math_fontfamily="cm")
    time_label = ax.text(0.98, 0.96, "", transform=ax.transAxes,
                         ha="right", va="top", fontsize=10)

    def update(frame: int):
        cart_x = z[frame] - cart_width / 2
        pivot_x = z[frame]
        rod1_dx = x1[frame] - pivot_x
        rod1_dy = y1[frame]
        rod2_dx = x2[frame] - x1[frame]
        rod2_dy = y2[frame] - y1[frame]

        def offset_midpoint(x_start, y_start, dx, dy, offset):
            length = np.hypot(dx, dy)
            normal_x = -dy / length
            normal_y = dx / length
            return (
                x_start + dx / 2 + offset * normal_x,
                y_start + dy / 2 + offset * normal_y,
            )

        def offset_point(x_point, y_point, dx, dy, offset):
            length = np.hypot(dx, dy)
            return (
                x_point - offset * dy / length,
                y_point + offset * dx / length,
            )

        rod1.set_data([pivot_x, x1[frame]], [0, y1[frame]])
        rod2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        rail.set_data([ax.get_xlim()[0], ax.get_xlim()[1]],
                      [ground_y, ground_y])
        ground.set_data([ax.get_xlim()[0], ax.get_xlim()[1]],
                        [ground_y, ground_y])
        pivot.center = (pivot_x, 0)
        bob1.center = (x1[frame], y1[frame])
        bob2.center = (x2[frame], y2[frame])
        cart.set_xy((cart_x, -cart_height))
        wheel1.center = (pivot_x - cart_width * 0.25, -cart_height - wheel_radius)
        wheel2.center = (pivot_x + cart_width * 0.25, -cart_height - wheel_radius)
        bob1_label.set_position(offset_point(x1[frame], y1[frame],
                                             rod1_dx, rod1_dy, 0.23))
        bob2_label.set_position(offset_point(x2[frame], y2[frame],
                                             rod2_dx, rod2_dy, 0.23))
        rod1_label.set_position(offset_midpoint(pivot_x, 0, rod1_dx, rod1_dy, 0.18))
        rod2_label.set_position(offset_midpoint(x1[frame], y1[frame],
                                                rod2_dx, rod2_dy, 0.18))
        time_label.set_text(f"t = {t[frame]:.2f} s")

        return (rod1, rod2, rail, ground, pivot, bob1, bob2, cart, wheel1,
                wheel2, bob1_label, bob2_label, rod1_label, rod2_label,
                time_label, *ground_hatching)

    if fps is None:
        if t.size > 1:
            dt = float(np.median(np.diff(t)))
            fps = max(1, round(1 / dt)) if dt > 0 else 30
        else:
            fps = 30

    animation = FuncAnimation(
        fig,
        update,
        frames=states.shape[0],
        interval=1000 / fps,
        # Full redraws prevent rods and labels from the previous frame from
        # remaining visible when the pendulum crosses the cart or rail.
        blit=False,
    )

    if save_path is not None:
        animation.save(str(save_path), fps=fps)
    if show:
        plt.show()

    return fig, animation

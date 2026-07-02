"""
visualizer.py — All the figures of the project.

Figure 1 — Original image + gradient map
Figure 2 — Cost map with the optimal path overlaid
Figure 3 — Dijkstra vs naive path (Bresenham) comparison

Each function accepts an optional `save_path` argument to export the
figure at high resolution (300 dpi PNG).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _path_to_arrays(
    path: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Splits a list of (row, col) into two numpy arrays."""
    rows = np.array([p[0] for p in path], dtype=float)
    cols = np.array([p[1] for p in path], dtype=float)
    return rows, cols


def _mark_point(ax: plt.Axes, pixel: tuple[int, int], color: str, label: str) -> None:
    """Draws an annotated marker on a pixel."""
    ax.plot(pixel[1], pixel[0], "o", color=color, markersize=9, zorder=5)
    ax.annotate(
        label,
        xy=(pixel[1], pixel[0]),
        xytext=(pixel[1] + 5, pixel[0] - 5),
        fontsize=8,
        color=color,
        fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Figure 1 — Original image + gradient map
# ---------------------------------------------------------------------------

def figure_image_and_gradient(
    gray: np.ndarray,
    gradient: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Displays the grayscale image and its gradient map side by side.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image [0, 1].
    gradient : np.ndarray
        Gradient map (output of cost_map.compute_gradient_map).
    save_path : str, optional
        If provided, saves the figure to this path (PNG, 300 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Original image and gradient map", fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Grayscale image")
    axes[0].axis("off")

    im = axes[1].imshow(gradient, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Gradient map (Sobel)")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("|gradient I|  (normalized)", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Cost map + optimal path
# ---------------------------------------------------------------------------

def figure_cost_and_path(
    cost_map: np.ndarray,
    path: list[tuple[int, int]],
    src_pixel: tuple[int, int],
    dst_pixel: tuple[int, int],
    waypoints: list[tuple[int, int]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlays the optimal path on the cost map.

    Parameters
    ----------
    cost_map : np.ndarray
        Cost map (0=edge, 1=homogeneous interior).
    path : list of (row, col)
        Optimal path (Dijkstra).
    src_pixel, dst_pixel : (row, col)
        Start and end points.
    waypoints : list of (row, col), optional
        Intermediate points, if any.
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Cost map and optimal path (Dijkstra)", fontsize=13, fontweight="bold")

    im = ax.imshow(cost_map, cmap="viridis", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cost  w = exp(-alpha * |gradient I|)", fontsize=9)

    # Draw the path
    rows, cols = _path_to_arrays(path)
    ax.plot(cols, rows, color="red", linewidth=1.5, label="Dijkstra path", zorder=4)

    # Source / destination points
    _mark_point(ax, src_pixel, "#00FF88", "S")
    _mark_point(ax, dst_pixel, "#FF6600", "D")

    # Intermediate waypoints
    if waypoints:
        for i, wp in enumerate(waypoints):
            _mark_point(ax, wp, "#FFD700", f"W{i+1}")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Path length: {len(path)} pixels", fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Dijkstra vs naive path comparison
# ---------------------------------------------------------------------------

def figure_comparison(
    gray: np.ndarray,
    path_dijkstra: list[tuple[int, int]],
    path_naive: list[tuple[int, int]],
    src_pixel: tuple[int, int],
    dst_pixel: tuple[int, int],
    cost_dijkstra: float,
    cost_naive: float,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Compares the Dijkstra path and the naive path (Bresenham straight line)
    overlaid on the original image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image [0, 1].
    path_dijkstra, path_naive : list of (row, col)
        The two paths to compare.
    src_pixel, dst_pixel : (row, col)
    cost_dijkstra, cost_naive : float
        Respective cumulative costs.
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "Comparison: Dijkstra vs straight line (Bresenham)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, path, title, color, cost in [
        (axes[0], path_dijkstra, "Dijkstra (optimal path)", "#FF4040", cost_dijkstra),
        (axes[1], path_naive,   "Straight line (naive)",    "#4080FF", cost_naive),
    ]:
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)

        rows, cols = _path_to_arrays(path)
        ax.plot(cols, rows, color=color, linewidth=2.0, label=title)

        _mark_point(ax, src_pixel, "#00FF88", "S")
        _mark_point(ax, dst_pixel, "#FF8800", "D")

        ax.set_title(f"{title}\nCumulative cost: {cost:.2f}  |  {len(path)} pixels", fontsize=9)
        ax.axis("off")

    # Cost reduction ratio
    if cost_naive > 0:
        reduction = (cost_naive - cost_dijkstra) / cost_naive * 100
        fig.text(
            0.5, 0.01,
            f"Cost reduction: {reduction:.1f}%",
            ha="center", fontsize=10, style="italic", color="#444444",
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved -> {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Quick overlay on an existing figure (used by main.py)
# ---------------------------------------------------------------------------

def overlay_path_on_image(
    ax: plt.Axes,
    path: list[tuple[int, int]],
    color: str = "red",
    linewidth: float = 2.0,
    label: str = "Path",
) -> None:
    """
    Draws a path on an existing matplotlib Axes.

    Parameters
    ----------
    ax : plt.Axes
    path : list of (row, col)
    color : str
    linewidth : float
    label : str
    """
    rows, cols = _path_to_arrays(path)
    ax.plot(cols, rows, color=color, linewidth=linewidth, label=label, zorder=4)

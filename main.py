"""
main.py — Interactive entry point of the project.

Usage
-----
    python main.py --image data/your_image.png

The user clicks two points on the displayed image.
The optimal path (Dijkstra) is instantly overlaid in red.

Options
-------
    --image PATH        Path to the input image (default: data/sample.png)
    --sigma FLOAT       Smoothing parameter for the gradient (default: 1.5)
    --alpha FLOAT       Edge sensitivity: cost = exp(-alpha * grad) (default: 8)
    --connectivity INT  4 or 8 (default: 8)
    --waypoints         Enables multi-point mode (click >= 2 points,
                        press Enter to compute the final path)
    --save-figures DIR  Directory where the 3 figures are saved (default: None)
    --no-compare        Disables Figure 3 (comparison)

Example
-------
    python main.py --image data/brain_mri.png --sigma 2 --waypoints --save-figures results/
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Ensures the src modules are importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.cost_map import load_image, compute_gradient_map, compute_cost_map
from src.graph_builder import build_graph
from src.pathfinder import find_path, find_path_multipoint, straight_line_path, path_cost
from src.visualizer import (
    figure_image_and_gradient,
    figure_cost_and_path,
    figure_comparison,
    overlay_path_on_image,
)


# ---------------------------------------------------------------------------
# Mouse / keyboard interface
# ---------------------------------------------------------------------------

class ClickCollector:
    """Collects the user's clicks on a matplotlib Axes."""

    def __init__(self, ax: plt.Axes, label: str = "point") -> None:
        self.ax = ax
        self.label = label
        self.points: list[tuple[int, int]] = []  # (row, col)
        self._cid = ax.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        if event.inaxes != self.ax or event.button != 1:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        self.points.append((row, col))
        self.ax.plot(col, row, "x", color="cyan", markersize=10, markeredgewidth=2, zorder=6)
        self.ax.figure.canvas.draw()
        print(f"  Click {len(self.points)}: pixel ({row}, {col})")

    def disconnect(self) -> None:
        self.ax.figure.canvas.mpl_disconnect(self._cid)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run(
    image_path: str,
    sigma: float = 1.5,
    alpha: float = 8.0,
    grad_threshold: float = 0.0,
    connectivity: int = 8,
    waypoints_mode: bool = False,
    save_dir: str | None = None,
    compare: bool = True,
) -> None:
    print("\n=== Loading and preparing the image ===")
    gray = load_image(image_path)
    H, W = gray.shape
    print(f"  Image loaded: {W}x{H} pixels")

    gradient = compute_gradient_map(gray, sigma=sigma)
    cost_map  = compute_cost_map(gray, sigma=sigma, alpha=alpha, grad_threshold=grad_threshold)
    print(f"  Cost map computed (sigma={sigma}, alpha={alpha}, threshold={grad_threshold})")

    # --- Figure 1 ---
    save1 = os.path.join(save_dir, "fig1_gradient.png") if save_dir else None
    fig1 = figure_image_and_gradient(gray, gradient, save_path=save1)

    print("\n=== Building the graph ===")
    graph = build_graph(cost_map, connectivity=connectivity)
    print(f"  Graph built: {graph.shape[0]} nodes, {connectivity}-connectivity")

    # --- Interactive interface to collect points ---
    print("\n=== Interactive point selection ===")
    if waypoints_mode:
        print("  Click >= 2 points on the image, then close the window.")
    else:
        print("  Click exactly 2 points (source, destination), then close the window.")

    fig_pick, ax_pick = plt.subplots(figsize=(8, 8))
    ax_pick.imshow(gray, cmap="gray")
    ax_pick.set_title(
        "Click the start/end points (close the window to confirm)",
        fontsize=10,
    )
    ax_pick.axis("off")

    collector = ClickCollector(ax_pick, label="waypoint")
    plt.show(block=True)
    collector.disconnect()

    clicked = collector.points
    if len(clicked) < 2:
        print("  Error: at least 2 clicks are required. Stopping.")
        return

    src = clicked[0]
    dst = clicked[-1]
    intermediate = clicked[1:-1] if len(clicked) > 2 else []

    # --- Path computation ---
    print("\n=== Computing the shortest path ===")
    all_waypoints = [src] + intermediate + [dst]

    if len(all_waypoints) == 2 or not waypoints_mode:
        # Simple case: source -> destination
        path_opt = find_path(graph, src, dst, (H, W))
    else:
        # Multi-point mode
        path_opt = find_path_multipoint(graph, all_waypoints, (H, W))

    path_straight = straight_line_path(src, dst)

    c_opt    = path_cost(cost_map, path_opt)
    c_naive  = path_cost(cost_map, path_straight)

    print(f"  Dijkstra path : {len(path_opt)} pixels | cost = {c_opt:.3f}")
    print(f"  Naive path    : {len(path_straight)} pixels | cost = {c_naive:.3f}")

    # --- Figure 2 ---
    save2 = os.path.join(save_dir, "fig2_path.png") if save_dir else None
    fig2 = figure_cost_and_path(
        cost_map, path_opt, src, dst,
        waypoints=intermediate if intermediate else None,
        save_path=save2,
    )

    # --- Figure 3 ---
    if compare:
        save3 = os.path.join(save_dir, "fig3_comparison.png") if save_dir else None
        fig3 = figure_comparison(
            gray, path_opt, path_straight,
            src, dst, c_opt, c_naive,
            save_path=save3,
        )

    plt.show(block=True)
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shortest-path segmentation in an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", default="data/sample.png",
        help="Path to the input image."
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5,
        help="Gaussian smoothing parameter (default: 1.5)."
    )
    parser.add_argument(
        "--alpha", type=float, default=8.0,
        help="Edge sensitivity: cost = exp(-alpha * grad) (default: 8)."
    )
    parser.add_argument(
        "--grad-threshold", type=float, default=0.0, dest="grad_threshold",
        help="Gradient threshold [0,1]: ignores weak edges such as shadows (default: 0.0)."
    )
    parser.add_argument(
        "--connectivity", type=int, choices=[4, 8], default=8,
        help="Graph connectivity: 4 or 8 (default: 8)."
    )
    parser.add_argument(
        "--waypoints", action="store_true",
        help="Enables intermediate multi-point mode."
    )
    parser.add_argument(
        "--save-figures", metavar="DIR", default=None,
        help="Directory where figures are saved."
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Disables the Dijkstra vs naive comparison figure."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.save_figures:
        os.makedirs(args.save_figures, exist_ok=True)

    run(
        image_path=args.image,
        sigma=args.sigma,
        alpha=args.alpha,
        grad_threshold=args.grad_threshold,
        connectivity=args.connectivity,
        waypoints_mode=args.waypoints,
        save_dir=args.save_figures,
        compare=not args.no_compare,
    )

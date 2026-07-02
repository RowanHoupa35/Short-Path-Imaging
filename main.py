"""
main.py — Point d'entrée interactif du projet.

Usage
-----
    python main.py --image data/your_image.png

L'utilisateur clique deux points sur l'image affichée.
Le chemin optimal (Dijkstra) s'affiche instantanément en rouge.

Options
-------
    --image PATH        Chemin vers l'image d'entrée (défaut: data/sample.png)
    --sigma FLOAT       Paramètre de lissage pour le gradient (défaut: 1.5)
    --alpha FLOAT       Sensibilité aux bords : coût = exp(-alpha * grad) (défaut: 8)
    --connectivity INT  4 ou 8 (défaut: 8)
    --waypoints         Active le mode multi-points (cliquer ≥ 2 points,
                        appuyer sur Entrée pour calculer le chemin final)
    --save-figures DIR  Répertoire où sauvegarder les 3 figures (défaut: None)
    --no-compare        Désactive la Figure 3 (comparaison)

Exemple
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

# Assure l'import des modules src( depuis la racine du projet
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
# Interface clavier / souris
# ---------------------------------------------------------------------------

class ClickCollector:
    """Collecte les clics de l'utilisateur sur un Axes matplotlib."""

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
        print(f"  Clic {len(self.points)} : pixel ({row}, {col})")

    def disconnect(self) -> None:
        self.ax.figure.canvas.mpl_disconnect(self._cid)


# ---------------------------------------------------------------------------
# Flux principal
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
    print("\n=== Chargement et préparation de l'image ===")
    gray = load_image(image_path)
    H, W = gray.shape
    print(f"  Image chargée : {W}×{H} pixels")

    gradient = compute_gradient_map(gray, sigma=sigma)
    cost_map  = compute_cost_map(gray, sigma=sigma, alpha=alpha, grad_threshold=grad_threshold)
    print(f"  Carte de coûts calculée (σ={sigma}, α={alpha}, seuil={grad_threshold})")

    # --- Figure 1 ---
    save1 = os.path.join(save_dir, "fig1_gradient.png") if save_dir else None
    fig1 = figure_image_and_gradient(gray, gradient, save_path=save1)

    print("\n=== Construction du graphe ===")
    graph = build_graph(cost_map, connectivity=connectivity)
    print(f"  Graphe créé : {graph.shape[0]} nœuds, connexité-{connectivity}")

    # --- Interface interactive pour recueillir les points ---
    print("\n=== Sélection interactive des points ===")
    if waypoints_mode:
        print("  Cliquez ≥ 2 points sur l'image, puis fermez la fenêtre.")
    else:
        print("  Cliquez exactement 2 points (source, destination), puis fermez la fenêtre.")

    fig_pick, ax_pick = plt.subplots(figsize=(8, 8))
    ax_pick.imshow(gray, cmap="gray")
    ax_pick.set_title(
        "Cliquez les points de départ/arrivée (fermez pour valider)",
        fontsize=10,
    )
    ax_pick.axis("off")

    collector = ClickCollector(ax_pick, label="waypoint")
    plt.show(block=True)
    collector.disconnect()

    clicked = collector.points
    if len(clicked) < 2:
        print("  Erreur : il faut au moins 2 clics. Arrêt.")
        return

    src = clicked[0]
    dst = clicked[-1]
    intermediate = clicked[1:-1] if len(clicked) > 2 else []

    # --- Calcul des chemins ---
    print("\n=== Calcul du plus court chemin ===")
    all_waypoints = [src] + intermediate + [dst]

    if len(all_waypoints) == 2 or not waypoints_mode:
        # Cas simple : source → destination
        path_opt = find_path(graph, src, dst, (H, W))
    else:
        # Mode multi-points
        path_opt = find_path_multipoint(graph, all_waypoints, (H, W))

    path_straight = straight_line_path(src, dst)

    c_opt    = path_cost(cost_map, path_opt)
    c_naive  = path_cost(cost_map, path_straight)

    print(f"  Chemin Dijkstra : {len(path_opt)} pixels | coût = {c_opt:.3f}")
    print(f"  Chemin naïf     : {len(path_straight)} pixels | coût = {c_naive:.3f}")

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
    print("\nTerminé.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmentation par plus court chemin dans une image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", default="data/sample.png",
        help="Chemin vers l'image d'entrée."
    )
    parser.add_argument(
        "--sigma", type=float, default=1.5,
        help="Paramètre de lissage gaussien (défaut: 1.5)."
    )
    parser.add_argument(
        "--alpha", type=float, default=8.0,
        help="Sensibilité aux bords : coût = exp(-alpha * grad) (défaut: 8)."
    )
    parser.add_argument(
        "--grad-threshold", type=float, default=0.0, dest="grad_threshold",
        help="Seuil de gradient [0,1] : ignore les bords faibles comme les ombres (défaut: 0.0)."
    )
    parser.add_argument(
        "--connectivity", type=int, choices=[4, 8], default=8,
        help="Connexité du graphe : 4 ou 8 (défaut: 8)."
    )
    parser.add_argument(
        "--waypoints", action="store_true",
        help="Active le mode multi-points intermédiaires."
    )
    parser.add_argument(
        "--save-figures", metavar="DIR", default=None,
        help="Répertoire de sauvegarde des figures."
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Désactive la figure de comparaison Dijkstra vs naïf."
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

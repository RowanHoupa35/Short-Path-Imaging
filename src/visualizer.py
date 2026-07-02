"""
visualizer.py — Toutes les figures du projet.

Figure 1 — Image originale + carte de gradient
Figure 2 — Carte de coûts avec chemin optimal superposé
Figure 3 — Comparaison Dijkstra vs chemin naïf (Bresenham)

Chaque fonction accepte un argument `save_path` (optionnel) pour exporter
la figure en haute résolution (300 dpi PNG).
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Utilitaires bas niveau
# ---------------------------------------------------------------------------

def _path_to_arrays(
    path: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sépare une liste de (row, col) en deux tableaux numpy."""
    rows = np.array([p[0] for p in path], dtype=float)
    cols = np.array([p[1] for p in path], dtype=float)
    return rows, cols


def _mark_point(ax: plt.Axes, pixel: tuple[int, int], color: str, label: str) -> None:
    """Trace un marker annoté sur un pixel."""
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
# Figure 1 — Image originale + carte de gradient
# ---------------------------------------------------------------------------

def figure_image_and_gradient(
    gray: np.ndarray,
    gradient: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Affiche l'image en niveaux de gris et sa carte de gradient côte à côte.

    Parameters
    ----------
    gray : np.ndarray
        Image en niveaux de gris [0, 1].
    gradient : np.ndarray
        Carte de gradient (sortie de cost_map.compute_gradient_map).
    save_path : str, optional
        Si fourni, sauvegarde la figure à ce chemin (PNG, 300 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Image originale et carte de gradient", fontsize=14, fontweight="bold")

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Image en niveaux de gris")
    axes[0].axis("off")

    im = axes[1].imshow(gradient, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Carte de gradient (Sobel)")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("|∇I|  (normalisé)", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure sauvegardée → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Carte de coûts + chemin optimal
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
    Superpose le chemin optimal à la carte de coûts.

    Parameters
    ----------
    cost_map : np.ndarray
        Carte de coûts (0=bord, 1=intérieur homogène).
    path : list of (row, col)
        Chemin optimal (Dijkstra).
    src_pixel, dst_pixel : (row, col)
        Points de départ et d'arrivée.
    waypoints : list of (row, col), optional
        Points intermédiaires, s'il y en a.
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Carte de coûts et chemin optimal (Dijkstra)", fontsize=13, fontweight="bold")

    im = ax.imshow(cost_map, cmap="viridis", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coût  w = 1/(1+|∇I|)", fontsize=9)

    # Tracé du chemin
    rows, cols = _path_to_arrays(path)
    ax.plot(cols, rows, color="red", linewidth=1.5, label="Chemin Dijkstra", zorder=4)

    # Points source / destination
    _mark_point(ax, src_pixel, "#00FF88", "S")
    _mark_point(ax, dst_pixel, "#FF6600", "D")

    # Waypoints intermédiaires
    if waypoints:
        for i, wp in enumerate(waypoints):
            _mark_point(ax, wp, "#FFD700", f"W{i+1}")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Chemin de longueur {len(path)} pixels", fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure sauvegardée → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Comparaison Dijkstra vs chemin naïf
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
    Compare le chemin Dijkstra et le chemin naïf (ligne droite de Bresenham)
    superposés sur l'image originale.

    Parameters
    ----------
    gray : np.ndarray
        Image en niveaux de gris [0, 1].
    path_dijkstra, path_naive : list of (row, col)
        Les deux chemins à comparer.
    src_pixel, dst_pixel : (row, col)
    cost_dijkstra, cost_naive : float
        Coûts cumulatifs respectifs.
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "Comparaison : Dijkstra vs ligne droite (Bresenham)",
        fontsize=13,
        fontweight="bold",
    )

    for ax, path, title, color, cost in [
        (axes[0], path_dijkstra, "Dijkstra (chemin optimal)", "#FF4040", cost_dijkstra),
        (axes[1], path_naive,   "Ligne droite (naïf)",        "#4080FF", cost_naive),
    ]:
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)

        rows, cols = _path_to_arrays(path)
        ax.plot(cols, rows, color=color, linewidth=2.0, label=title)

        _mark_point(ax, src_pixel, "#00FF88", "S")
        _mark_point(ax, dst_pixel, "#FF8800", "D")

        ax.set_title(f"{title}\nCoût cumulatif : {cost:.2f}  |  {len(path)} pixels", fontsize=9)
        ax.axis("off")

    # Ratio de réduction de coût
    if cost_naive > 0:
        reduction = (cost_naive - cost_dijkstra) / cost_naive * 100
        fig.text(
            0.5, 0.01,
            f"Réduction de coût : {reduction:.1f} %",
            ha="center", fontsize=10, style="italic", color="#444444",
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure sauvegardée → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Overlay rapide sur une figure existante (utilisé par main.py)
# ---------------------------------------------------------------------------

def overlay_path_on_image(
    ax: plt.Axes,
    path: list[tuple[int, int]],
    color: str = "red",
    linewidth: float = 2.0,
    label: str = "Chemin",
) -> None:
    """
    Trace un chemin sur un Axes matplotlib existant.

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

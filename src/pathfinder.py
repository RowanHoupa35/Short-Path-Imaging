"""
pathfinder.py — Recherche du plus court chemin dans le graphe de pixels.

Deux stratégies sont proposées :
  1. scipy.sparse.csgraph.dijkstra   — rapide, vectorisé, recommandé
  2. networkx.dijkstra_path           — plus lisible, utile pour les petites images

La fonction principale `find_path` retourne la liste ordonnée des coordonnées
(ligne, colonne) du chemin optimal entre deux pixels.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from src.graph_builder import pixel_to_node, node_to_pixel


def find_path(
    graph: csr_matrix,
    src_pixel: tuple[int, int],
    dst_pixel: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Calcule le plus court chemin (Dijkstra) entre deux pixels.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix, shape (N, N)
        Graphe de grille pondéré (sortie de graph_builder.build_graph).
    src_pixel : (row, col)
        Pixel de départ.
    dst_pixel : (row, col)
        Pixel d'arrivée.
    image_shape : (H, W)
        Dimensions de l'image (nécessaires pour la conversion nœud ↔ pixel).

    Returns
    -------
    list of (row, col)
        Séquence ordonnée de pixels formant le chemin optimal.

    Raises
    ------
    ValueError
        Si aucun chemin n'existe entre src et dst.
    """
    H, W = image_shape
    src_node = pixel_to_node(*src_pixel, W)
    dst_node = pixel_to_node(*dst_pixel, W)

    # Dijkstra depuis le nœud source uniquement (directed=False → graphe non orienté)
    dist_matrix, predecessors = scipy_dijkstra(
        graph,
        directed=False,
        indices=src_node,
        return_predecessors=True,
    )

    if np.isinf(dist_matrix[dst_node]):
        raise ValueError(
            f"Aucun chemin entre {src_pixel} et {dst_pixel}. "
            "Vérifiez la connectivité du graphe."
        )

    # Reconstruction du chemin par remontée des prédécesseurs
    path_nodes = _reconstruct_path(predecessors, src_node, dst_node)
    path_pixels = [node_to_pixel(n, W) for n in path_nodes]
    return path_pixels


def find_path_multipoint(
    graph: csr_matrix,
    waypoints: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Calcule le chemin optimal passant par une liste ordonnée de waypoints.

    Enchaîne les appels à `find_path` entre chaque paire de points consécutifs.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
    waypoints : list of (row, col)
        Au moins deux points. Le chemin reliera waypoints[0] → waypoints[1] → …
    image_shape : (H, W)

    Returns
    -------
    list of (row, col)
        Chemin concaténé (sans duplication des jonctions).
    """
    if len(waypoints) < 2:
        raise ValueError("Au moins deux points sont nécessaires.")

    full_path: list[tuple[int, int]] = []
    for i in range(len(waypoints) - 1):
        segment = find_path(graph, waypoints[i], waypoints[i + 1], image_shape)
        if i == 0:
            full_path.extend(segment)
        else:
            full_path.extend(segment[1:])  # évite la duplication du point de jonction

    return full_path


def straight_line_path(
    src_pixel: tuple[int, int],
    dst_pixel: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Retourne les pixels du segment droit (Bresenham) entre src et dst.
    Utilisé comme chemin de référence "naïf" pour la comparaison.
    """
    r0, c0 = src_pixel
    r1, c1 = dst_pixel

    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return points


def path_cost(
    cost_map: np.ndarray,
    path: list[tuple[int, int]],
) -> float:
    """
    Calcule le coût total d'un chemin comme la somme des coûts des pixels traversés.

    Parameters
    ----------
    cost_map : np.ndarray, shape (H, W)
    path : list of (row, col)

    Returns
    -------
    float
    """
    return float(sum(cost_map[r, c] for r, c in path))


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _reconstruct_path(
    predecessors: np.ndarray,
    src: int,
    dst: int,
) -> list[int]:
    """Remonte le tableau des prédécesseurs de dst jusqu'à src."""
    path = []
    current = dst
    while current != src:
        if current < 0:
            raise ValueError("Chemin introuvable — prédécesseur invalide.")
        path.append(current)
        current = predecessors[current]
    path.append(src)
    path.reverse()
    return path

"""
graph_builder.py — Construction du graphe de grille à partir de la carte de coûts.

Chaque pixel (i, j) devient un nœud d'indice linéaire i*W + j.
Des arêtes pondérées relient chaque pixel à ses 8 voisins (4-connexité ou 8-connexité).
Le graphe est stocké sous forme de matrice creuse CSR (scipy.sparse) pour limiter
la consommation mémoire sur de grandes images.

Convention de pondération :
    w(u → v) = (cost[u] + cost[v]) / 2
Cela rend le coût symétrique et prend en compte les deux extrémités de chaque arête.
"""

import numpy as np
from scipy import sparse


# Décalages des voisins, une seule direction par paire d'arête non-orientée
# (le graphe est utilisé avec directed=False dans scipy_dijkstra, qui accepte
# indifféremment csr[i, j] ou csr[j, i] — stocker les deux sens doublerait
# la mémoire et le temps de construction pour rien).
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

_NEIGHBORS_4 = [(-1, 0), (0, -1)]


def build_graph(
    cost_map: np.ndarray,
    connectivity: int = 8,
    diagonal_weight: float = 1.4142,  # √2 pour les arêtes diagonales
) -> sparse.csr_matrix:
    """
    Construit le graphe de grille pondéré à partir d'une carte de coûts.

    Parameters
    ----------
    cost_map : np.ndarray, shape (H, W)
        Valeur de coût de chaque pixel (float, dans (0, 1]).
    connectivity : int
        4 (connexité-4) ou 8 (connexité-8, par défaut).
    diagonal_weight : float
        Facteur multiplicatif pour les arêtes diagonales (≈ √2 pour respecter
        la distance euclidienne entre centres de pixels adjacents en diagonale).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (N, N)
        Matrice d'adjacence creuse (N = H * W).
        G[u, v] = coût de la transition u → v.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity doit être 4 ou 8.")

    H, W = cost_map.shape
    N = H * W

    neighbors = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4

    rows = []
    cols = []
    data = []

    # Pré-aplatissement pour accéder rapidement aux coûts
    cost_flat = cost_map.ravel()

    for dr, dc in neighbors:
        is_diag = (dr != 0 and dc != 0)

        # Indices source : tous les pixels
        r_src = np.arange(H)
        c_src = np.arange(W)
        R_src, C_src = np.meshgrid(r_src, c_src, indexing="ij")

        # Indices destination
        R_dst = R_src + dr
        C_dst = C_src + dc

        # Masque des pixels dont le voisin est dans l'image
        valid = (
            (R_dst >= 0) & (R_dst < H) &
            (C_dst >= 0) & (C_dst < W)
        )

        u = (R_src[valid] * W + C_src[valid]).ravel()
        v = (R_dst[valid] * W + C_dst[valid]).ravel()

        # Coût de l'arête = moyenne des coûts des deux pixels extrémités
        edge_cost = (cost_flat[u] + cost_flat[v]) / 2.0

        # Pondération supplémentaire pour les arêtes diagonales
        if is_diag:
            edge_cost = edge_cost * diagonal_weight

        rows.append(u)
        cols.append(v)
        data.append(edge_cost)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    graph = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return graph


def pixel_to_node(r: int, c: int, W: int) -> int:
    """Convertit les coordonnées (ligne, colonne) d'un pixel en indice de nœud."""
    return r * W + c


def node_to_pixel(node: int, W: int) -> tuple[int, int]:
    """Convertit un indice de nœud en coordonnées (ligne, colonne)."""
    return divmod(node, W)

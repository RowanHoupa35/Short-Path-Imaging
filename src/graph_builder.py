"""
graph_builder.py — Builds the grid graph from the cost map.

Each pixel (i, j) becomes a node with linear index i*W + j.
Weighted edges connect each pixel to its neighbors (4-connectivity or 8-connectivity).
The graph is stored as a sparse CSR matrix (scipy.sparse) to limit memory
consumption on large images.

Weighting convention:
    w(u -> v) = (cost[u] + cost[v]) / 2
This makes the cost symmetric and accounts for both endpoints of each edge.
"""

import numpy as np
from scipy import sparse


# Neighbor offsets, one direction per undirected edge pair
# (the graph is used with directed=False in scipy_dijkstra, which accepts
# either csr[i, j] or csr[j, i] — storing both directions would double
# memory and build time for no benefit).
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

_NEIGHBORS_4 = [(-1, 0), (0, -1)]


def build_graph(
    cost_map: np.ndarray,
    connectivity: int = 8,
    diagonal_weight: float = 1.4142,  # sqrt(2) for diagonal edges
) -> sparse.csr_matrix:
    """
    Builds the weighted grid graph from a cost map.

    Parameters
    ----------
    cost_map : np.ndarray, shape (H, W)
        Cost value of each pixel (float, in (0, 1]).
    connectivity : int
        4 (4-connectivity) or 8 (8-connectivity, default).
    diagonal_weight : float
        Multiplicative factor for diagonal edges (~= sqrt(2) to respect
        the Euclidean distance between diagonally adjacent pixel centers).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (N, N)
        Sparse adjacency matrix (N = H * W).
        G[u, v] = cost of the transition u -> v.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    H, W = cost_map.shape
    N = H * W

    neighbors = _NEIGHBORS_8 if connectivity == 8 else _NEIGHBORS_4

    rows = []
    cols = []
    data = []

    # Pre-flattening for fast access to the costs
    cost_flat = cost_map.ravel()

    for dr, dc in neighbors:
        is_diag = (dr != 0 and dc != 0)

        # Source indices: all pixels
        r_src = np.arange(H)
        c_src = np.arange(W)
        R_src, C_src = np.meshgrid(r_src, c_src, indexing="ij")

        # Destination indices
        R_dst = R_src + dr
        C_dst = C_src + dc

        # Mask of pixels whose neighbor lies within the image
        valid = (
            (R_dst >= 0) & (R_dst < H) &
            (C_dst >= 0) & (C_dst < W)
        )

        u = (R_src[valid] * W + C_src[valid]).ravel()
        v = (R_dst[valid] * W + C_dst[valid]).ravel()

        # Edge cost = average of the costs of both endpoint pixels
        edge_cost = (cost_flat[u] + cost_flat[v]) / 2.0

        # Extra weighting for diagonal edges
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
    """Converts pixel (row, col) coordinates to a node index."""
    return r * W + c


def node_to_pixel(node: int, W: int) -> tuple[int, int]:
    """Converts a node index to (row, col) pixel coordinates."""
    return divmod(node, W)

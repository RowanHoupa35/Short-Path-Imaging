"""
pathfinder.py — Shortest-path search in the pixel graph.

Two strategies are available:
  1. scipy.sparse.csgraph.dijkstra   — fast, vectorized, recommended
  2. networkx.dijkstra_path           — more readable, useful for small images

The main function `find_path` returns the ordered list of (row, col)
coordinates of the optimal path between two pixels.
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
    Computes the shortest path (Dijkstra) between two pixels.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix, shape (N, N)
        Weighted grid graph (output of graph_builder.build_graph).
    src_pixel : (row, col)
        Starting pixel.
    dst_pixel : (row, col)
        Destination pixel.
    image_shape : (H, W)
        Image dimensions (needed for node <-> pixel conversion).

    Returns
    -------
    list of (row, col)
        Ordered sequence of pixels forming the optimal path.

    Raises
    ------
    ValueError
        If no path exists between src and dst.
    """
    H, W = image_shape
    src_node = pixel_to_node(*src_pixel, W)
    dst_node = pixel_to_node(*dst_pixel, W)

    # Dijkstra from the source node only (directed=False -> undirected graph)
    dist_matrix, predecessors = scipy_dijkstra(
        graph,
        directed=False,
        indices=src_node,
        return_predecessors=True,
    )

    if np.isinf(dist_matrix[dst_node]):
        raise ValueError(
            f"No path between {src_pixel} and {dst_pixel}. "
            "Check the graph connectivity."
        )

    # Reconstruct the path by walking back through the predecessors
    path_nodes = _reconstruct_path(predecessors, src_node, dst_node)
    path_pixels = [node_to_pixel(n, W) for n in path_nodes]
    return path_pixels


def find_path_multipoint(
    graph: csr_matrix,
    waypoints: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Computes the optimal path passing through an ordered list of waypoints.

    Chains calls to `find_path` between each pair of consecutive points.

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
    waypoints : list of (row, col)
        At least two points. The path connects waypoints[0] -> waypoints[1] -> ...
    image_shape : (H, W)

    Returns
    -------
    list of (row, col)
        Concatenated path (without duplicating the junction points).
    """
    if len(waypoints) < 2:
        raise ValueError("At least two points are required.")

    full_path: list[tuple[int, int]] = []
    for i in range(len(waypoints) - 1):
        segment = find_path(graph, waypoints[i], waypoints[i + 1], image_shape)
        if i == 0:
            full_path.extend(segment)
        else:
            full_path.extend(segment[1:])  # avoids duplicating the junction point

    return full_path


def straight_line_path(
    src_pixel: tuple[int, int],
    dst_pixel: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Returns the pixels of the straight (Bresenham) segment between src and dst.
    Used as the "naive" reference path for comparison.
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
    Computes the total cost of a path as the sum of the costs of the
    pixels it traverses.

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
# Internal helpers
# ---------------------------------------------------------------------------

def _reconstruct_path(
    predecessors: np.ndarray,
    src: int,
    dst: int,
) -> list[int]:
    """Walks the predecessor array back from dst to src."""
    path = []
    current = dst
    while current != src:
        if current < 0:
            raise ValueError("Path not found — invalid predecessor.")
        path.append(current)
        current = predecessors[current]
    path.append(src)
    path.reverse()
    return path

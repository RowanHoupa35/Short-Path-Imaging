import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from src.graph_builder import build_graph, pixel_to_node, node_to_pixel


def test_pixel_to_node_and_back_roundtrip():
    W = 7
    for r, c in [(0, 0), (3, 4), (6, 6)]:
        node = pixel_to_node(r, c, W)
        assert node_to_pixel(node, W) == (r, c)


def test_pixel_to_node_matches_row_major_layout():
    W = 5
    assert pixel_to_node(0, 0, W) == 0
    assert pixel_to_node(0, 4, W) == 4
    assert pixel_to_node(1, 0, W) == 5
    assert pixel_to_node(2, 3, W) == 13


def test_build_graph_invalid_connectivity_raises():
    cost_map = np.ones((3, 3))
    with pytest.raises(ValueError):
        build_graph(cost_map, connectivity=6)


def test_build_graph_shape():
    H, W = 4, 5
    cost_map = np.ones((H, W))
    graph = build_graph(cost_map, connectivity=4)
    assert graph.shape == (H * W, H * W)


def test_build_graph_4_connectivity_edge_count_no_duplicates():
    H, W = 4, 5
    cost_map = np.ones((H, W))
    graph = build_graph(cost_map, connectivity=4)

    # Chaque arête non-orientée doit être stockée une seule fois :
    # H*(W-1) arêtes horizontales + (H-1)*W arêtes verticales.
    expected_edges = H * (W - 1) + (H - 1) * W
    assert graph.nnz == expected_edges


def test_build_graph_8_connectivity_edge_count_no_duplicates():
    H, W = 4, 5
    cost_map = np.ones((H, W))
    graph = build_graph(cost_map, connectivity=8)

    horizontal_vertical = H * (W - 1) + (H - 1) * W
    diagonal = 2 * (H - 1) * (W - 1)
    expected_edges = horizontal_vertical + diagonal
    assert graph.nnz == expected_edges


def test_build_graph_edge_weight_is_average_of_endpoint_costs():
    cost_map = np.array([[0.2, 0.8], [0.4, 0.6]])
    graph = build_graph(cost_map, connectivity=4)

    u = pixel_to_node(0, 0, W=2)
    v = pixel_to_node(0, 1, W=2)
    expected_weight = (cost_map[0, 0] + cost_map[0, 1]) / 2.0

    # L'arête n'est stockée que dans un seul sens (voir commentaire dans
    # graph_builder.py) : on accepte indifféremment graph[u, v] ou graph[v, u].
    stored_weight = graph[u, v] if graph[u, v] != 0 else graph[v, u]
    assert stored_weight == pytest.approx(expected_weight)


def test_build_graph_diagonal_weight_applied():
    cost_map = np.full((2, 2), 0.5)
    graph = build_graph(cost_map, connectivity=8, diagonal_weight=1.4142)

    u = pixel_to_node(0, 0, W=2)
    v = pixel_to_node(1, 1, W=2)  # voisin diagonal
    expected_weight = ((0.5 + 0.5) / 2.0) * 1.4142

    stored_weight = graph[u, v] if graph[u, v] != 0 else graph[v, u]
    assert stored_weight == pytest.approx(expected_weight)


def test_build_graph_undirected_dijkstra_is_symmetric():
    """Même en ne stockant chaque arête que dans un sens, scipy_dijkstra
    avec directed=False doit donner des distances symétriques."""
    rng = np.random.default_rng(0)
    cost_map = rng.uniform(0.1, 1.0, size=(6, 6))
    graph = build_graph(cost_map, connectivity=8)

    node_a = pixel_to_node(0, 0, W=6)
    node_b = pixel_to_node(5, 5, W=6)

    dist_from_a = scipy_dijkstra(graph, directed=False, indices=node_a)
    dist_from_b = scipy_dijkstra(graph, directed=False, indices=node_b)

    assert dist_from_a[node_b] == pytest.approx(dist_from_b[node_a])

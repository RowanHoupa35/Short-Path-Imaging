import numpy as np
import pytest
from scipy import sparse

from src.graph_builder import build_graph
from src.pathfinder import (
    find_path,
    find_path_multipoint,
    straight_line_path,
    path_cost,
)


def test_straight_line_path_horizontal():
    path = straight_line_path((0, 0), (0, 4))
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_straight_line_path_vertical():
    path = straight_line_path((0, 0), (4, 0))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_straight_line_path_diagonal():
    path = straight_line_path((0, 0), (3, 3))
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_straight_line_path_endpoints_included():
    src, dst = (2, 7), (9, 1)
    path = straight_line_path(src, dst)
    assert path[0] == src
    assert path[-1] == dst


def test_path_cost_sums_pixel_values():
    cost_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = [(0, 0), (0, 1), (1, 1)]
    assert path_cost(cost_map, path) == pytest.approx(1.0 + 2.0 + 4.0)


def test_find_path_uniform_cost_map_is_straight_line():
    H, W = 5, 1
    cost_map = np.ones((H, W))
    graph = build_graph(cost_map, connectivity=4)

    path = find_path(graph, (0, 0), (4, 0), (H, W))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_find_path_avoids_high_cost_region():
    """Le chemin optimal doit contourner un mur de coût élevé plutôt que
    le traverser en ligne droite, si un détour moins coûteux existe."""
    H, W = 5, 5
    cost_map = np.ones((H, W))
    cost_map[2, :] = 50.0  # mur horizontal coûteux au milieu, sauf un passage
    cost_map[2, 4] = 1.0   # ouverture à droite

    graph = build_graph(cost_map, connectivity=4)
    path = find_path(graph, (0, 0), (4, 0), (H, W))

    # Le chemin optimal ne doit pas traverser le mur coûteux directement
    # sous la colonne de départ.
    assert (2, 0) not in path
    straight = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    assert path_cost(cost_map, path) < path_cost(cost_map, straight)


def test_find_path_raises_when_unreachable():
    # Graphe manuel à 2 nœuds sans arête entre eux.
    graph = sparse.csr_matrix((2, 2))
    with pytest.raises(ValueError):
        find_path(graph, (0, 0), (0, 1), image_shape=(1, 2))


def test_find_path_multipoint_avoids_duplicate_junction():
    H, W = 1, 5
    cost_map = np.ones((H, W))
    graph = build_graph(cost_map, connectivity=4)

    waypoints = [(0, 0), (0, 2), (0, 4)]
    path = find_path_multipoint(graph, waypoints, (H, W))

    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    # Le point de jonction (0, 2) ne doit apparaître qu'une seule fois.
    assert path.count((0, 2)) == 1


def test_find_path_multipoint_requires_at_least_two_waypoints():
    graph = build_graph(np.ones((2, 2)), connectivity=4)
    with pytest.raises(ValueError):
        find_path_multipoint(graph, [(0, 0)], (2, 2))

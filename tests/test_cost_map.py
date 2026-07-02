import numpy as np
import pytest

from src.cost_map import compute_gradient_map, compute_cost_map, load_image


def _step_image(size: int = 20) -> np.ndarray:
    """Image avec un bord net vertical au milieu (moitié noire, moitié blanche)."""
    img = np.zeros((size, size), dtype=float)
    img[:, size // 2:] = 1.0
    return img


def test_compute_gradient_map_range_and_shape():
    img = _step_image()
    grad = compute_gradient_map(img, sigma=1.0)

    assert grad.shape == img.shape
    assert grad.min() >= 0.0
    assert grad.max() <= 1.0 + 1e-9


def test_compute_gradient_map_uniform_image_has_zero_gradient():
    img = np.full((10, 10), 0.5)
    grad = compute_gradient_map(img, sigma=1.0)

    assert np.allclose(grad, 0.0)


def test_compute_cost_map_uniform_image_has_cost_one():
    img = np.full((10, 10), 0.5)
    cost = compute_cost_map(img, sigma=1.0, alpha=8.0)

    assert np.allclose(cost, 1.0)


def test_compute_cost_map_range():
    img = _step_image()
    cost = compute_cost_map(img, sigma=1.0, alpha=8.0)

    assert cost.min() > 0.0
    assert cost.max() <= 1.0 + 1e-9


def test_compute_cost_map_edge_has_lower_cost_than_interior():
    img = _step_image(size=40)
    cost = compute_cost_map(img, sigma=1.0, alpha=8.0)

    edge_cost = cost[20, 20]        # sur le bord
    interior_cost = cost[20, 5]     # zone homogène

    assert edge_cost < interior_cost


def test_grad_threshold_flattens_weak_gradients():
    img = _step_image(size=40)
    grad = compute_gradient_map(img, sigma=1.0)
    weak_threshold = grad.max() + 0.01  # seuil au-dessus du gradient le plus fort

    cost_no_threshold = compute_cost_map(img, sigma=1.0, alpha=8.0, grad_threshold=0.0)
    cost_thresholded = compute_cost_map(img, sigma=1.0, alpha=8.0, grad_threshold=weak_threshold)

    # Avec un seuil supérieur à tous les gradients, plus aucun bord n'est détecté
    # donc le coût doit être uniformément 1.0.
    assert np.allclose(cost_thresholded, 1.0)
    # Sans seuillage, il existe bien un bord de coût plus faible.
    assert cost_no_threshold.min() < 1.0


def test_grad_threshold_zero_is_noop():
    img = _step_image()
    cost_default = compute_cost_map(img, sigma=1.0, alpha=8.0)
    cost_explicit_zero = compute_cost_map(img, sigma=1.0, alpha=8.0, grad_threshold=0.0)

    assert np.array_equal(cost_default, cost_explicit_zero)


def test_load_image_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_image("data/does_not_exist.png")


def test_load_image_returns_normalized_grayscale(tmp_path):
    from skimage.io import imsave

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[:, 8:] = 255
    path = tmp_path / "synthetic.png"
    imsave(str(path), rgb)

    gray = load_image(str(path))

    assert gray.shape == (16, 16)
    assert gray.min() >= 0.0
    assert gray.max() <= 1.0

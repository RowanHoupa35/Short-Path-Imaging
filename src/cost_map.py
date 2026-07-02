"""
cost_map.py — Computes the cost map from the image gradient.

The cost of a pixel is w = exp(-alpha * |gradient I|), i.e.:
  - cost close to 0  -> strong gradient (edge)
  - cost close to 1  -> weak gradient (homogeneous interior)

Dijkstra will therefore favor paths that follow the edges.
"""

import numpy as np
import cv2
from skimage.filters import sobel, gaussian
from skimage.color import rgb2gray
from skimage import img_as_float


def load_image(path: str) -> np.ndarray:
    """
    Loads an image and converts it to grayscale float [0, 1].

    Parameters
    ----------
    path : str
        Path to the image (JPEG, PNG, TIFF...).

    Returns
    -------
    np.ndarray
        Grayscale image, values in [0, 1], shape (H, W).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # BGR -> RGB conversion if needed, then grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = rgb2gray(img_as_float(img))
    else:
        gray = img_as_float(img)

    return gray


def compute_gradient_map(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Computes the gradient map (Sobel magnitude) after Gaussian smoothing.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image [0, 1].
    sigma : float
        Standard deviation of the pre-smoothing Gaussian filter.

    Returns
    -------
    np.ndarray
        Gradient map normalized to [0, 1], shape (H, W).
    """
    smoothed = gaussian(gray, sigma=sigma)
    grad = sobel(smoothed)

    # Normalize to [0, 1]
    g_min, g_max = grad.min(), grad.max()
    if g_max > g_min:
        grad = (grad - g_min) / (g_max - g_min)

    return grad


def compute_cost_map(
    gray: np.ndarray,
    sigma: float = 1.0,
    alpha: float = 8.0,
    grad_threshold: float = 0.0,
) -> np.ndarray:
    """
    Computes the cost map: w(i,j) = exp(-alpha * |gradient I(i,j)|).

    Pixels on strong edges have a cost close to 0 (~e^-alpha),
    homogeneous interior pixels have a cost close to 1.0.
    The edge/interior contrast is exponential (~e^alpha times), which
    forces Dijkstra to follow contours even over long distances.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image [0, 1].
    sigma : float
        Smoothing parameter before computing the gradient.
    alpha : float
        Controls the penalty for edge-free areas. Recommended value: 5-10.
    grad_threshold : float
        Threshold in [0, 1]: normalized gradients below this threshold
        are set to 0 before the cost is computed, which neutralizes
        weak edges (shadows, noise) by giving them a cost close to 1
        instead of a slight advantage. 0.0 disables thresholding (default).

    Returns
    -------
    np.ndarray
        Cost map in (0, 1], shape (H, W).
    """
    grad = compute_gradient_map(gray, sigma=sigma)
    if grad_threshold > 0:
        grad = np.where(grad < grad_threshold, 0.0, grad)
    cost = np.exp(-alpha * grad)
    return cost.astype(np.float64)

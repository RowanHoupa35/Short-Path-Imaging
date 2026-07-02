"""
cost_map.py — Calcul de la carte de coûts à partir du gradient de l'image.

Le coût d'un pixel est w = exp(-alpha * |∇I|), soit :
  - coût proche de 0  → fort gradient (bord)
  - coût proche de 1  → faible gradient (intérieur homogène)

Dijkstra favorisera donc les chemins qui longent les bords.
"""

import numpy as np
import cv2
from skimage.filters import sobel, gaussian
from skimage.color import rgb2gray
from skimage import img_as_float


def load_image(path: str) -> np.ndarray:
    """
    Charge une image et la convertit en niveaux de gris float [0, 1].

    Parameters
    ----------
    path : str
        Chemin vers l'image (JPEG, PNG, TIFF…).

    Returns
    -------
    np.ndarray
        Image en niveaux de gris, valeurs dans [0, 1], shape (H, W).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {path}")

    # Conversion BGR → RGB si nécessaire, puis niveaux de gris
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = rgb2gray(img_as_float(img))
    else:
        gray = img_as_float(img)

    return gray


def compute_gradient_map(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Calcule la carte de gradient (magnitude Sobel) après lissage gaussien.

    Parameters
    ----------
    gray : np.ndarray
        Image en niveaux de gris [0, 1].
    sigma : float
        Écart-type du filtre gaussien de pré-lissage.

    Returns
    -------
    np.ndarray
        Carte de gradient normalisée dans [0, 1], shape (H, W).
    """
    smoothed = gaussian(gray, sigma=sigma)
    grad = sobel(smoothed)

    # Normalisation dans [0, 1]
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
    Calcule la carte de coûts : w(i,j) = exp(-alpha * |∇I(i,j)|).

    Les pixels sur des bords forts ont un coût proche de 0 (~e^-alpha),
    les pixels intérieurs homogènes ont un coût proche de 1.0.
    Le contraste bord/intérieur est exponentiel (~e^alpha fois), ce qui
    force Dijkstra à longer les contours même sur de longues distances.

    Parameters
    ----------
    gray : np.ndarray
        Image en niveaux de gris [0, 1].
    sigma : float
        Paramètre de lissage avant calcul du gradient.
    alpha : float
        Contrôle la pénalité des zones sans bord. Valeur recommandée : 5–10.
    grad_threshold : float
        Seuil dans [0, 1] : les gradients normalisés inférieurs à ce seuil
        sont ramenés à 0 avant le calcul du coût, ce qui neutralise les
        bords faibles (ombres, bruit) en leur donnant un coût proche de 1
        au lieu d'un léger avantage. 0.0 désactive le seuillage (défaut).

    Returns
    -------
    np.ndarray
        Carte de coûts dans (0, 1], shape (H, W).
    """
    grad = compute_gradient_map(gray, sigma=sigma)
    if grad_threshold > 0:
        grad = np.where(grad < grad_threshold, 0.0, grad)
    cost = np.exp(-alpha * grad)
    return cost.astype(np.float64)

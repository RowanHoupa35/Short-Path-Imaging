"""
generate_test_images.py
Génère les images de test :
  - data/mri/   : 10 tranches T1 réelles (skimage.data.brain)
  - data/fruits/: téléchargement Wikimedia Commons + fallback synthétique (PIL)
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
from skimage import data as skdata
from skimage.io import imsave

ROOT = Path(__file__).parent / "data"
ROOT_MRI = ROOT / "mri"
ROOT_FRUITS = ROOT / "fruits"
ROOT_MRI.mkdir(parents=True, exist_ok=True)
ROOT_FRUITS.mkdir(parents=True, exist_ok=True)

# ── MRI ──────────────────────────────────────────────────────────────────────
# skimage.data.brain() : 10 coupes axiales T1 d'un cerveau humain (256×256, uint8)
print("=== IRM (skimage.data.brain) ===")
brain_vol = skdata.brain()   # shape (10, 256, 256)
for i, slc in enumerate(brain_vol):
    out = ROOT_MRI / f"brain_axial_{i:02d}.png"
    imsave(str(out), slc)
    print(f"  {out.name}  {slc.shape}  ✓")

# ── Fruits ────────────────────────────────────────────────────────────────────
FRUIT_URLS = {
    "apple.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg",
    "banana.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Chocolate-Chip-Cookies-101.jpg",
    "orange.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg",
    "strawberry.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/2/29/PerfectStrawberry.jpg",
    "lemon.jpg":
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Lemons.jpg",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"}

print("\n=== Fruits (Wikimedia Commons) ===")
for fname, url in FRUIT_URLS.items():
    dest = ROOT_FRUITS / fname
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r, open(dest, "wb") as f:
            data = r.read()
            f.write(data)
        size_kb = dest.stat().st_size // 1024
        if size_kb < 2:
            raise ValueError(f"Fichier trop petit ({size_kb} Ko) — probablement une erreur HTML")
        print(f"  {dest.name}  {size_kb} Ko  ✓")
    except Exception as exc:
        print(f"  {dest.name}  ECHEC ({exc})")
        # Fallback : image synthétique colorée avec un disque (test de robustesse)
        _img = np.ones((256, 256, 3), dtype=np.uint8) * 240  # fond clair
        rr, cc = np.ogrid[:256, :256]
        mask = (rr - 128) ** 2 + (cc - 128) ** 2 < 90 ** 2
        # arbre de couleur selon le fruit
        colors = {
            "apple.jpg":      [200,  30,  30],
            "banana.jpg":     [230, 220,  30],
            "orange.jpg":     [230, 120,  20],
            "strawberry.jpg": [210,  40,  60],
            "lemon.jpg":      [230, 230,  50],
        }
        _img[mask] = colors.get(fname, [150, 150, 150])
        imsave(str(dest), _img)
        print(f"    → image synthétique générée comme fallback")

print("\nTerminé.")
print(f"  IRM    : {len(list(ROOT_MRI.iterdir()))} fichiers dans {ROOT_MRI}")
print(f"  Fruits : {len(list(ROOT_FRUITS.iterdir()))} fichiers dans {ROOT_FRUITS}")

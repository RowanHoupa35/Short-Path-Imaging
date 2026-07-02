# Shortest-Path Imaging — Contour Segmentation via Graph Optimization

> **Context**: This project implements object contour segmentation in images by modeling the problem as a **shortest path in a pixel graph**. This approach is directly inspired by graph-based optimization segmentation methods used in medical imaging (MRI, retinal imaging).

---

## Core idea

Each pixel $(i, j)$ of an image is modeled as a **node** in a grid graph. Two neighboring pixels are connected by an edge whose **weight decays exponentially with the local gradient**:

$$w(u, v) = e^{-\alpha \, |\nabla I|}$$

A strong gradient signals an **edge** between two regions, so its cost is low. Dijkstra finds the minimum-cost path between two pixels clicked by the user — this path naturally follows the object's contours.

This mechanism is a special case of **Live Wire / Intelligent Scissors** methods, used in medical annotation software such as ITK-SNAP.

---

## Project structure

```
shortest-path-imaging/
├── data/                    # Test images (fruits, MRI) — see generate_test_images.py
├── src/
│   ├── cost_map.py          # Image loading + gradient map + cost map
│   ├── graph_builder.py     # Weighted grid graph (scipy.sparse CSR)
│   ├── pathfinder.py        # Dijkstra, multi-waypoints, naive reference path
│   └── visualizer.py        # Figures 1, 2, 3
├── tests/                   # Pytest suite (cost_map, graph_builder, pathfinder)
├── main.py                  # Interactive interface (mouse clicks)
├── generate_test_images.py  # (Re)generates the images in data/
├── conftest.py               # Pytest config (makes src/ importable)
├── requirements.txt
├── requirements-dev.txt     # requirements.txt + pytest
├── LICENSE
└── README.md
```

---

## Installation

Requirements: Python ≥ 3.9.

```bash
git clone https://github.com/RowanHoupa35/Short-Path-Imaging.git
cd Short-Path-Imaging

# Virtual environment (venv or conda), then:
pip install -r requirements.txt

# To run the tests, also install the dev dependencies:
pip install -r requirements-dev.txt
```

Test images are already included in `data/`. To regenerate them (or add more via `skimage.data` / Wikimedia Commons):

```bash
python generate_test_images.py
```

---

## Usage

> There is no default image: `--image` is required.

```bash
# Launch the interactive interface (click 2 points, close the window to confirm)
python main.py --image data/fruits/apple.jpg

# With stronger smoothing and figure export
python main.py --image data/mri/brain_axial_00.png --sigma 2.0 --save-figures results/

# Multi-waypoint mode (contour guided by several intermediate points)
python main.py --image data/fruits/orange.jpg --waypoints

# 4-connectivity only (no diagonals)
python main.py --image data/fruits/lemon.jpg --connectivity 4

# Ignore weak edges (shadows, noise) below this gradient threshold [0, 1]
python main.py --image data/mri/brain_axial_03.png --grad-threshold 0.15
```

---

## Generated visualizations

| Figure | Description |
|--------|-------------|
| **Fig 1** | Original image + Sobel gradient map |
| **Fig 2** | Cost map $w = e^{-\alpha \|\nabla I\|}$ with the Dijkstra path overlaid |
| **Fig 3** | Dijkstra vs straight line (Bresenham) comparison with cumulative costs |

---

## Algorithmic pipeline

```
Image (PNG/JPEG/TIFF)
    │
    ▼ load_image()          → grayscale [0, 1]
    │
    ▼ compute_gradient_map()→ Sobel after Gaussian smoothing (σ)
    │
    ▼ compute_cost_map()    → w = exp(-α · |∇I|)
    │
    ▼ build_graph()         → sparse CSR matrix (4- or 8-connectivity)
    │
    ▼ dijkstra()            → scipy.sparse.csgraph.shortest_path
    │
    ▼ find_path()           → list of pixels (row, col)
    │
    ▼ visualizer            → 3 high-resolution figures
```

---

## Tests

Pytest suite covering `cost_map`, `graph_builder`, and `pathfinder` (value ranges, edge counts, edge cases, Dijkstra's behavior around a high-cost region, etc.):

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Connection to graph-based medical segmentation

This implementation is a practical introduction to graph-based segmentation methods (Graph Cuts, Random Walker, Shortest Path) widely used in medical imaging:

- **Intelligent Scissors** (Mortensen & Barrett, 1995) — interactive segmentation used in many annotation tools.
- **Graph Cuts** (Boykov & Jolly, 2001) — organ segmentation via min-cut / max-flow.
- **Random Walker** (Grady, 2006) — probabilistic segmentation via the graph Laplace equation.

The conceptual bridge is direct: minimizing a cost over a pixel graph amounts to solving a discrete optimization problem on a grid structure — the foundation of graph-based medical segmentation approaches.

---

## Suggested test images

| Image | Source | Purpose |
|-------|--------|---------|
| Fruit photo | local | Quick algorithm validation |
| Brain MRI slice | [BraTS](https://www.med.upenn.edu/cbica/brats/) | Medical segmentation |
| Retina image | [DRIVE dataset](https://drive.grand-challenge.org/) | Vascular segmentation |

---

## Author

Rowan Houpa — Personal ML project (March 2026)

---

## License

Distributed under the MIT License — see [LICENSE](LICENSE).

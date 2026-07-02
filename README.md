# Shortest-Path Imaging — Segmentation de contours par optimisation sur graphe

> **Contexte** : Ce projet implémente une segmentation de contours d'objets dans des images en modélisant le problème comme un **plus court chemin dans un graphe de pixels**. Cette approche est directement inspirée des méthodes de segmentation par optimisation sur graphes utilisées en imagerie médicale (IRM, rétinographie).

---

## Idée intuitive

Chaque pixel $(i, j)$ d'une image est modélisé comme un **nœud** dans un graphe de grille. Deux pixels voisins sont reliés par une arête dont le **poids décroît exponentiellement avec le gradient local** :

$$w(u, v) = e^{-\alpha \, |\nabla I|}$$

Un gradient élevé signale un **bord** entre deux régions ; son coût est donc faible. Dijkstra trouve le chemin de coût minimal entre deux pixels cliqués par l'utilisateur — ce chemin longe naturellement les contours de l'objet.

Ce mécanisme est un cas particulier des méthodes de **Live Wire / Intelligent Scissors**, utilisées dans des logiciels d'annotation médicale comme ITK-SNAP.

---

## Structure du projet

```
shortest-path-imaging/
├── data/                    # Images de test (fruits, IRM) — voir generate_test_images.py
├── src/
│   ├── cost_map.py          # Chargement image + carte de gradient + carte de coûts
│   ├── graph_builder.py     # Graphe de grille pondéré (scipy.sparse CSR)
│   ├── pathfinder.py        # Dijkstra, multi-waypoints, chemin naïf de référence
│   └── visualizer.py        # Figures 1, 2, 3
├── tests/                   # Suite pytest (cost_map, graph_builder, pathfinder)
├── main.py                  # Interface interactive (clic souris)
├── generate_test_images.py  # (Re)génère les images de data/
├── conftest.py               # Config pytest (rend src/ importable)
├── requirements.txt
├── requirements-dev.txt     # requirements.txt + pytest
├── LICENSE
└── README.md
```

---

## Installation

Prérequis : Python ≥ 3.9.

```bash
git clone https://github.com/<votre-username>/shortest-path-imaging.git
cd shortest-path-imaging

# Environnement virtuel (venv ou conda), puis :
pip install -r requirements.txt

# Pour lancer les tests, installez aussi les dépendances de dev :
pip install -r requirements-dev.txt
```

Les images de test sont déjà incluses dans `data/`. Pour les régénérer (ou en
ajouter d'autres via `skimage.data` / Wikimedia Commons) :

```bash
python generate_test_images.py
```

---

## Utilisation

> Il n'y a pas d'image par défaut : `--image` est obligatoire.

```bash
# Lancer l'interface interactive (cliquer 2 points, fermer la fenêtre pour valider)
python main.py --image data/fruits/apple.jpg

# Avec lissage plus fort et sauvegarde des figures
python main.py --image data/mri/brain_axial_00.png --sigma 2.0 --save-figures results/

# Mode multi-waypoints (contour guidé par plusieurs points intermédiaires)
python main.py --image data/fruits/orange.jpg --waypoints

# Connexité-4 uniquement (pas de diagonales)
python main.py --image data/fruits/lemon.jpg --connectivity 4

# Ignore les bords faibles (ombres, bruit) sous ce seuil de gradient [0, 1]
python main.py --image data/mri/brain_axial_03.png --grad-threshold 0.15
```

---

## Visualisations produites

| Figure | Description |
|--------|-------------|
| **Fig 1** | Image originale + carte de gradient Sobel |
| **Fig 2** | Carte de coûts $w = e^{-\alpha \|\nabla I\|}$ avec chemin Dijkstra superposé |
| **Fig 3** | Comparaison Dijkstra vs ligne droite (Bresenham) avec coûts cumulatifs |

---

## Pipeline algorithmique

```
Image (PNG/JPEG/TIFF)
    │
    ▼ load_image()          → niveaux de gris [0, 1]
    │
    ▼ compute_gradient_map()→ Sobel après lissage gaussien (σ)
    │
    ▼ compute_cost_map()    → w = exp(-α · |∇I|)
    │
    ▼ build_graph()         → matrice creuse CSR (connexité 4 ou 8)
    │
    ▼ dijkstra()            → scipy.sparse.csgraph.shortest_path
    │
    ▼ find_path()           → liste de pixels (row, col)
    │
    ▼ visualizer            → 3 figures haute résolution
```

---

## Tests

Suite pytest couvrant `cost_map`, `graph_builder` et `pathfinder` (plages de
valeurs, comptage d'arêtes, cas limites, comportement de Dijkstra face à une
zone de coût élevé, etc.) :

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Lien avec la segmentation médicale par optimisation sur graphes

Cette implémentation constitue une introduction pratique aux méthodes de segmentation par graphes (Graph Cut, Random Walker, Shortest Path) omniprésentes en imagerie médicale :

- **Intelligent Scissors** (Mortensen & Barrett, 1995) — segmentation interactive utilisée dans de nombreux outils d'annotation.
- **Graph Cuts** (Boykov & Jolly, 2001) — segmentation d'organes par min-cut / max-flow.
- **Random Walker** (Grady, 2006) — segmentation probabiliste par équation de Laplace sur graphe.

Le pont conceptuel est direct : minimiser un coût sur un graphe de pixels revient à résoudre un problème d'optimisation discrète sur une structure de grille — c'est la base des approches de segmentation du Pr. Talbot (ESIEE / LIGM).

---

## Images de test suggérées

| Image | Source | Utilité |
|-------|--------|---------|
| Photo de fruit | locale | Validation rapide de l'algo |
| Coupe IRM cerveau | [BraTS](https://www.med.upenn.edu/cbica/brats/) | Segmentation médicale |
| Image de rétine | [DRIVE dataset](https://drive.grand-challenge.org/) | Segmentation vasculaire |

---

## Auteur

Rowan Houpa — Projet ML personnel (Mars 2026)

---

## Licence

Distribué sous licence MIT — voir [LICENSE](LICENSE).

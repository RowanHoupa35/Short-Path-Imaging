"""Racine pytest : garantit que `src` est importable en tant que package
depuis n'importe quel répertoire d'où pytest est lancé."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

"""Permette ai test di importare i moduli della cartella codice/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

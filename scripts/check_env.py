"""Vérifier que l'environnement du projet est prêt.

Usage : python scripts/check_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODULES = [
    "numpy", "pandas", "pyarrow", "duckdb", "scipy", "tqdm", "kagglehub",
    "requests", "dotenv", "kaggle", "sklearn", "implicit", "lightgbm",
    "sentence_transformers", "torch", "networkx", "matplotlib", "umap",
    "streamlit",
]


def main():
    print(f"Python {sys.version.split()[0]}")
    manquants = []
    for nom in MODULES:
        try:
            module = importlib.import_module(nom)
            print(f"  OK   {nom:<22} {getattr(module, '__version__', '')}")
        except ImportError as erreur:
            manquants.append(nom)
            print(f"  ERR  {nom:<22} {erreur}")

    import torch
    print(f"\nGPU CUDA disponible pour torch : {torch.cuda.is_available()}")

    import config
    print(f"Catalogue d'animes trouvé : {config.ANIME_CSV.exists()}")
    print(f"MAL_CLIENT_ID renseigné   : {bool(config.MAL_CLIENT_ID)}")

    if manquants:
        print("\nModules manquants : python -m pip install -r requirements.txt")
        sys.exit(1)
    print("\nEnvironnement prêt.")


if __name__ == "__main__":
    main()

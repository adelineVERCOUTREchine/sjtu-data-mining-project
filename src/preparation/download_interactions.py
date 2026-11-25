"""Télécharger le dataset Kaggle des listes MyAnimeList dans data/raw/.

Nécessite KAGGLE_USERNAME et KAGGLE_KEY dans .env (voir .env.example).
Archive d'environ 660 Mo ; environ 2,8 Go une fois décompressée.
"""

import shutil
from pathlib import Path

import kagglehub

import config

DESCRIPTION = "Télécharger les listes MAL (Kaggle, ~660 Mo) dans data/raw/"


def ajouter_arguments(parser):
    pass


def executer(args):
    destination = config.INTERACTIONS_RAW_DIR
    if destination.exists() and any(destination.iterdir()):
        print(f"Déjà présent : {destination}")
        return

    print(f"Téléchargement de {config.KAGGLE_INTERACTIONS}...")
    cache = Path(kagglehub.dataset_download(config.KAGGLE_INTERACTIONS))
    shutil.copytree(cache, destination, dirs_exist_ok=True)

    print(f"Fichiers copiés dans {destination} :")
    for fichier in sorted(destination.iterdir()):
        print(f"  {fichier.name:<30} {fichier.stat().st_size / 1e6:>10.1f} Mo")

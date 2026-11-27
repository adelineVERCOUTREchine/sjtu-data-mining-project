"""Chemins et paramètres partagés par tous les modules du projet."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

try:  # python-dotenv peut manquer sur Kaggle ; les clés y sont inutiles.
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

DATA_DIR = PROJECT_DIR / "data"

# Catalogue de 4 985 animes et fichiers dérivés (versionnés)
CONTENT_DIR = DATA_DIR / "content"
ANIME_CSV = CONTENT_DIR / "anime_dataset.csv"
SYNOPSIS_CSV = CONTENT_DIR / "anime_synopsis.csv"
SYNOPSIS_TRAIN = CONTENT_DIR / "anime_synopsis.train"
GENRES_TAGS_TRAIN = CONTENT_DIR / "anime_genres_tags.train"
GENRES_TAGS_SYNOPSIS_TRAIN = CONTENT_DIR / "anime_genres_tags_synopsis.train"

# Données volumineuses (non versionnées)
RAW_DIR = DATA_DIR / "raw"              # téléchargements bruts (Kaggle, API)
PROCESSED_DIR = Path(os.getenv("DMANIME_PROCESSED_DIR") or DATA_DIR / "processed")
MODELS_DIR = PROJECT_DIR / "models"     # modèles entraînés, embeddings

# Sorties des méthodes (versionnées)
RESULTS_DIR = PROJECT_DIR / "results"

# Listes MyAnimeList (voir docs/methodes_recommandation.md, section 2.1)
KAGGLE_INTERACTIONS = "hernan4444/anime-recommendation-database-2020"
# Sur Kaggle, le dataset est monté en lecture seule ailleurs : DMANIME_RAW_DIR le désigne.
INTERACTIONS_RAW_DIR = Path(os.getenv("DMANIME_RAW_DIR")
                            or RAW_DIR / KAGGLE_INTERACTIONS.split("/")[-1])

# Snapshot MyAnimeList jusqu'à octobre 2025 : fiches (details.csv), notes des
# utilisateurs (ratings.csv, 124 M lignes), recommandations MAL, profils.
KAGGLE_MAL2025 = "neelagiriaditya/anime-dataset-jan-1917-to-oct-2025"
MAL2025_DIR = Path(os.getenv("DMANIME_MAL2025_DIR") or RAW_DIR / "mal-2025" / "datasets")

# API
MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID", "")
MAL_REDIRECT_URI = os.getenv("MAL_REDIRECT_URI", "http://127.0.0.1:8501/")
MAL_CLIENT_SECRET = os.getenv("MAL_CLIENT_SECRET", "")
MAL_API_URL = "https://api.myanimelist.net/v2"
ANILIST_API_URL = "https://graphql.anilist.co"

SEED = 42

# Exclure les animes adultes (colonne is_adult : 63 animes classés Rx - Hentai)
# des recommandations, de l'évaluation et des cartes.
EXCLURE_ADULTES = True

for _dossier in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _dossier.mkdir(parents=True, exist_ok=True)

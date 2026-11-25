"""Chargement des matrices produites par preparation/prepare_interactions.py."""

from dataclasses import dataclass

import pandas as pd
import scipy.sparse as sp

import config


@dataclass
class Split:
    nom: str
    input: sp.csr_matrix    # animes visibles (ce que le modèle reçoit)
    target: sp.csr_matrix   # animes masqués à retrouver
    exclude: sp.csr_matrix  # déjà vus : jamais recommandés


def _charger(nom):
    chemin = config.PROCESSED_DIR / f"{nom}.npz"
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} absent : lancer d'abord prepare_interactions.")
    return sp.load_npz(chemin).tocsr()


def charger_train(notes=False):
    """Matrice utilisateurs d'entraînement x animes (binaire, ou notes MAL)."""
    return _charger("train_ratings" if notes else "train")


def charger_split(nom):
    if nom not in ("val", "test"):
        raise ValueError("Split attendu : val ou test.")
    return Split(nom, *(_charger(f"{nom}_{partie}") for partie in ("input", "target", "exclude")))


def charger_items():
    """DataFrame index -> mal_id, titre (même ordre que les colonnes des matrices)."""
    return pd.read_csv(config.PROCESSED_DIR / "items.csv")


def masque_tout_public(mal_ids):
    """Booléen par anime : False pour les animes adultes (is_adult) si EXCLURE_ADULTES."""
    import numpy as np
    if not config.EXCLURE_ADULTES:
        return np.ones(len(mal_ids), dtype=bool)
    adultes = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "is_adult"])
    adultes = set(adultes.mal_id[adultes.is_adult.astype(str).str.lower() == "true"])
    return ~pd.Series(mal_ids).isin(adultes).to_numpy()

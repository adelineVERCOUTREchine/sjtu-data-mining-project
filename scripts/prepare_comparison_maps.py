"""Préparer les cartes et les modèles compacts pour la comparaison hors ligne."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config
from ease_map import projeter
from methods.item_knn import entrainer
from series import CatalogueSeries


def main():
    with np.load(config.MODELS_DIR / "ease_demo.npz") as modele:
        ids = modele["mal_ids"]
        source = modele["source"].item()
    dossier = config.DATA_DIR / "processed" / str(source)
    if not (dossier / "items.csv").exists():
        dossier = config.PROCESSED_DIR
    if not np.array_equal(pd.read_csv(dossier / "items.csv").mal_id, ids):
        raise ValueError("Le catalogue des interactions doit correspondre au modèle EASE.")
    catalogue = CatalogueSeries(ids, config.CONTENT_DIR / "series.csv")
    sortie = config.MODELS_DIR / "comparison"
    sortie.mkdir(exist_ok=True)
    for nom in ("item_knn", "bge"):
        if nom == "item_knn":
            matrice = entrainer(sp.load_npz(dossier / "train.npz"))
            groupes, coords = projeter(matrice, catalogue)
            creuse = sp.csr_matrix(matrice)
            contenu = dict(data=creuse.data, indices=creuse.indices, indptr=creuse.indptr)
        else:
            with np.load(config.MODELS_DIR / "embeddings" / "bge-base-en-v1.5.npz") as cache:
                indices = pd.Index(cache["mal_id"]).get_indexer(ids)
                if np.any(indices < 0):
                    raise ValueError("Embeddings manquants pour certains titres.")
                vecteurs = cache["vecteurs"][indices]
            groupes, coords = projeter(vecteurs, catalogue, coefficients=False)
            contenu = dict(vecteurs=vecteurs)
        np.savez_compressed(sortie / f"{nom}.npz", ids=ids, series=catalogue.series, source=source,
                            groupes=groupes, coords=coords, **contenu)
        print(f"{nom} : {len(groupes)} séries", flush=True)


if __name__ == "__main__":
    main()

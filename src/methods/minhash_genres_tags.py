"""MinHash sur les genres et tags : paires d'animes au Jaccard estimé élevé."""

import csv
import time

import config
from common.minhash import (
    ajouter_arguments_minhash,
    creer_fonctions_hachage,
    creer_labels,
    exporter_similarites,
    signatures_vocabulaire_reduit,
    verifier_arguments_minhash,
)

DESCRIPTION = "MinHash + Jaccard sur les genres et tags (labels complets, k = 1)"
SHINGLE_SIZE = 1  # Un label entier, même s'il contient plusieurs mots.

calculer_signatures = signatures_vocabulaire_reduit


def charger_documents(source):
    documents = {}
    ignores = 0
    with source.open(encoding="utf-8-sig", newline="") as entree:
        for numero, ligne in enumerate(csv.reader(entree, delimiter="\t"), 1):
            if len(ligne) != 3 or not ligne[0].strip():
                raise ValueError(f"Ligne {numero} : attendu id, genres, tags (tabulations).")
            identifiant, genres, tags = ligne
            if identifiant in documents:
                raise ValueError(f"Identifiant en double : {identifiant}")
            labels = creer_labels(genres, tags)
            if not labels:
                ignores += 1
                continue
            documents[identifiant] = labels
    return documents, ignores


def ajouter_arguments(parser):
    ajouter_arguments_minhash(
        parser,
        entree=config.GENRES_TAGS_TRAIN,
        sortie=config.RESULTS_DIR / "minhash" / "prediction_genres_tags.csv",
        seuil=0.5,
    )


def executer(args):
    verifier_arguments_minhash(args)
    debut = time.perf_counter()
    documents, ignores = charger_documents(args.input)
    print(f"{len(documents)} animes, {ignores} ensembles vides ignorés.")
    moyenne = sum(map(len, documents.values())) / len(documents) if documents else 0
    print(f"k = {SHINGLE_SIZE} label ; {moyenne:.2f} shingles/anime ; {args.num_hashes} fonctions.")
    fonctions = creer_fonctions_hachage(args.num_hashes, args.seed)
    signatures = calculer_signatures(documents, fonctions)
    paires = exporter_similarites(documents, signatures, args.output, args.threshold)
    print(f"{paires} paires avec Jaccard estimé > {args.threshold} : {args.output}")
    print(f"Durée : {time.perf_counter() - debut:.2f} s.")

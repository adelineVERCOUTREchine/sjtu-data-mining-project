"""MinHash sur l'union des genres, tags et shingles de synopsis."""

import csv
import time

import config
from common.minhash import (
    ajouter_arguments_minhash,
    creer_fonctions_hachage,
    creer_labels,
    creer_shingles_mots,
    exporter_similarites,
    signatures_par_document,
    verifier_arguments_minhash,
)

DESCRIPTION = "MinHash + Jaccard sur genres/tags (k = 1) et synopsis (k = 3) combinés"
SYNOPSIS_SHINGLE_SIZE = 3

calculer_signatures = signatures_par_document


def creer_shingles(genres, tags, synopsis):
    # Le train contient déjà les synopsis nettoyés par prepare_content.
    texte = creer_shingles_mots(synopsis, SYNOPSIS_SHINGLE_SIZE, prefixe="synopsis:")
    return creer_labels(genres, tags) | texte


def charger_documents(source):
    documents = {}
    ignores = 0
    identifiants = set()
    with source.open(encoding="utf-8-sig", newline="") as entree:
        for numero, ligne in enumerate(csv.reader(entree, delimiter="\t"), 1):
            if len(ligne) != 4 or not ligne[0].strip():
                raise ValueError(
                    f"Ligne {numero} : attendu id, genres, tags, synopsis (tabulations)."
                )
            identifiant, genres, tags, synopsis = ligne
            if identifiant in identifiants:
                raise ValueError(f"Identifiant en double : {identifiant}")
            identifiants.add(identifiant)
            # Comparer le même ensemble d'animes que pour les synopsis seuls.
            if len(synopsis.split()) < SYNOPSIS_SHINGLE_SIZE:
                ignores += 1
                continue
            documents[identifiant] = creer_shingles(genres, tags, synopsis)
    return documents, ignores


def ajouter_arguments(parser):
    ajouter_arguments_minhash(
        parser,
        entree=config.GENRES_TAGS_SYNOPSIS_TRAIN,
        sortie=config.RESULTS_DIR / "minhash" / "prediction_genres_tags_synopsis.csv",
        seuil=0.1,
    )


def executer(args):
    verifier_arguments_minhash(args)
    debut = time.perf_counter()
    documents, ignores = charger_documents(args.input)
    print(f"{len(documents)} animes, {ignores} synopsis trop courts ignorés.", flush=True)
    moyenne = sum(map(len, documents.values())) / len(documents) if documents else 0
    print(f"Genres/tags : 1 label par shingle ; synopsis : {SYNOPSIS_SHINGLE_SIZE} mots.")
    print(f"{moyenne:.2f} shingles/anime ; {args.num_hashes} fonctions ; graine {args.seed}.")
    fonctions = creer_fonctions_hachage(args.num_hashes, args.seed)
    signatures = calculer_signatures(documents, fonctions)
    print("Signatures calculées. Comparaison des paires...", flush=True)
    paires = exporter_similarites(documents, signatures, args.output, args.threshold)
    print(f"{paires} paires avec Jaccard estimé > {args.threshold} : {args.output}")
    print(f"Durée : {time.perf_counter() - debut:.2f} s.")

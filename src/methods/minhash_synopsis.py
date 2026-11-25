"""MinHash sur les synopsis seuls (shingles de trois mots).

Reprend l'ancien `src/main.py` (exercice MinHash du cours) : mêmes shingles,
même hachage SHA-1 sur 32 bits, mêmes 128 fonctions et même format de sortie.
"""

import time

import config
from common.minhash import (
    ajouter_arguments_minhash,
    creer_fonctions_hachage,
    creer_shingles_mots,
    exporter_similarites,
    signatures_par_document,
    verifier_arguments_minhash,
)

DESCRIPTION = "MinHash + Jaccard sur les synopsis (shingles de 3 mots)"
SHINGLE_SIZE = 3


def charger_documents(source):
    """Lire `id synopsis` (séparés par une espace) et créer les shingles."""
    documents = {}
    ignores = 0
    with source.open(encoding="utf-8") as entree:
        for ligne in entree:
            identifiant, _, synopsis = ligne.rstrip("\r\n").partition(" ")
            if not identifiant:
                continue
            if identifiant in documents:
                raise ValueError(f"Identifiant en double : {identifiant}")
            shingles = creer_shingles_mots(synopsis, SHINGLE_SIZE)
            if not shingles:
                ignores += 1
                continue
            documents[identifiant] = shingles
    return documents, ignores


def ajouter_arguments(parser):
    ajouter_arguments_minhash(
        parser,
        entree=config.SYNOPSIS_TRAIN,
        sortie=config.RESULTS_DIR / "minhash" / "prediction_synopsis.csv",
        seuil=0.1,
    )


def executer(args):
    verifier_arguments_minhash(args)
    debut = time.perf_counter()
    documents, ignores = charger_documents(args.input)
    moyenne = sum(map(len, documents.values())) / len(documents) if documents else 0
    print(f"{len(documents)} animes, {ignores} synopsis trop courts ignorés.")
    print(f"k = {SHINGLE_SIZE} mots ; {moyenne:.2f} shingles/anime ; {args.num_hashes} fonctions.")
    fonctions = creer_fonctions_hachage(args.num_hashes, args.seed)
    signatures = signatures_par_document(documents, fonctions)
    print("Signatures calculées. Comparaison des paires...", flush=True)
    paires = exporter_similarites(documents, signatures, args.output, args.threshold)
    print(f"{paires} paires avec Jaccard estimé > {args.threshold} : {args.output}")
    print(f"Durée : {time.perf_counter() - debut:.2f} s.")

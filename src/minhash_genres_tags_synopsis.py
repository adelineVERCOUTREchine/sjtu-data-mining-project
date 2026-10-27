"""Comparer ensemble genres, tags et synopsis avec MinHash et Jaccard exact."""

import argparse
import csv
import time
from hashlib import sha1
from pathlib import Path

from minhash_common import HASH_SEED, NEXT_PRIME, NUM_HASHES, creer_fonctions_hachage
from minhash_genres_tags import creer_labels, exporter_similarites


PROJECT_DIR = Path(__file__).resolve().parent.parent
SYNOPSIS_SHINGLE_SIZE = 3


def creer_shingles(genres, tags, synopsis):
    # Le train contient déjà les synopsis nettoyés par dataset_modifier.py.
    mots = synopsis.split()
    texte = {
        "synopsis:" + " ".join(mots[i:i + SYNOPSIS_SHINGLE_SIZE])
        for i in range(len(mots) - SYNOPSIS_SHINGLE_SIZE + 1)
    }
    return creer_labels(genres, tags) | frozenset(texte)


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


def calculer_signatures(documents, fonctions):
    signatures = []
    for shingles in documents.values():
        if not shingles:
            raise ValueError("Un ensemble vide ne doit pas recevoir de signature MinHash.")
        # Traiter un anime à la fois : ne pas stocker 128 valeurs par shingle
        # pour tout le vocabulaire des synopsis.
        identifiants = {
            int.from_bytes(sha1(shingle.encode("utf-8")).digest()[:4], "big")
            for shingle in shingles
        }
        signatures.append(tuple(
            min((a * identifiant + b) % NEXT_PRIME for identifiant in identifiants)
            for a, b in fonctions
        ))
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=PROJECT_DIR / "archive" / "anime_genres_tags_synopsis.train")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_DIR / "data" / "prediction_genres_tags_synopsis.csv")
    parser.add_argument("--num-hashes", type=int, default=NUM_HASHES)
    parser.add_argument("--seed", type=int, default=HASH_SEED)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.num_hashes < 1 or not 0 <= args.threshold <= 1:
        parser.error("--num-hashes doit être positif et --threshold compris entre 0 et 1.")

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


if __name__ == "__main__":
    main()

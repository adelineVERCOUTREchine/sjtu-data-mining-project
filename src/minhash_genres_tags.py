"""Comparer les ensembles de genres et tags avec MinHash et Jaccard exact."""

import argparse
import csv
import time
import unicodedata
from hashlib import sha1
from pathlib import Path

from minhash_common import HASH_SEED, NEXT_PRIME, NUM_HASHES, creer_fonctions_hachage


PROJECT_DIR = Path(__file__).resolve().parent.parent
SHINGLE_SIZE = 1  # Un label entier, même s'il contient plusieurs mots.


def creer_labels(genres, tags):
    labels = set()
    for prefixe, champ in (("genre", genres), ("tag", tags)):
        for valeur in champ.split("|"):
            label = " ".join(unicodedata.normalize("NFKC", valeur).casefold().split())
            if label:
                labels.add(f"{prefixe}:{label}")
    return frozenset(labels)


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


def calculer_signatures(documents, fonctions):
    # Chaque label est haché une seule fois pour tout le corpus.
    valeurs = {}
    for labels in documents.values():
        for label in labels:
            if label not in valeurs:
                identifiant = int.from_bytes(sha1(label.encode("utf-8")).digest()[:4], "big")
                valeurs[label] = tuple((a * identifiant + b) % NEXT_PRIME for a, b in fonctions)

    cache = {}
    signatures = []
    for labels in documents.values():
        if not labels:
            raise ValueError("Un ensemble vide ne doit pas recevoir de signature MinHash.")
        if labels not in cache:
            cache[labels] = tuple(min(composantes) for composantes in zip(
                *(valeurs[label] for label in labels)
            ))
        signatures.append(cache[labels])
    return signatures


def exporter_similarites(documents, signatures, destination, seuil=0.5):
    if not 0 <= seuil <= 1:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")
    identifiants = list(documents)
    ensembles = list(documents.values())
    nombre_hashes = len(signatures[0]) if signatures else NUM_HASHES
    paires = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as sortie:
        ecrivain = csv.writer(sortie)
        ecrivain.writerow(["article1", "article2", "Est. J", "Act. J"])
        for i, signature in enumerate(signatures):
            for j in range(i + 1, len(signatures)):
                estimation = sum(a == b for a, b in zip(signature, signatures[j])) / nombre_hashes
                if estimation > seuil:
                    intersection = len(ensembles[i] & ensembles[j])
                    exact = intersection / (len(ensembles[i]) + len(ensembles[j]) - intersection)
                    ecrivain.writerow([identifiants[i], identifiants[j], estimation, exact])
                    paires += 1
            if (i + 1) % 500 == 0:
                print(f"Comparaison : {i + 1}/{len(signatures)} animes traités.", flush=True)
    return paires


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / "archive" / "anime_genres_tags.train")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "data" / "prediction_genres_tags.csv")
    parser.add_argument("--num-hashes", type=int, default=NUM_HASHES)
    parser.add_argument("--seed", type=int, default=HASH_SEED)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.num_hashes < 1 or not 0 <= args.threshold <= 1:
        parser.error("--num-hashes doit être positif et --threshold compris entre 0 et 1.")

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


if __name__ == "__main__":
    main()

"""Briques MinHash communes à toutes les méthodes de similarité par ensembles."""

import csv
import random
import unicodedata
from hashlib import sha1
from pathlib import Path


NUM_HASHES = 128
HASH_SEED = 42
NEXT_PRIME = 4294967311  # Premier supérieur aux identifiants SHA-1 sur 32 bits.


def creer_fonctions_hachage(nombre=NUM_HASHES, seed=HASH_SEED):
    """Coefficients (a, b) des fonctions h(x) = (a*x + b) % NEXT_PRIME."""
    if nombre < 1:
        raise ValueError("Le nombre de fonctions de hachage doit être positif.")
    rng = random.Random(seed)
    coefficients_a = rng.sample(range(1, NEXT_PRIME), nombre)
    return [(a, rng.randrange(NEXT_PRIME)) for a in coefficients_a]


def hacher_shingle(shingle):
    """Identifiant 32 bits d'un shingle (4 premiers octets de SHA-1)."""
    return int.from_bytes(sha1(shingle.encode("utf-8")).digest()[:4], "big")


def creer_labels(genres, tags):
    """Ensemble de labels complets normalisés, préfixés par leur champ."""
    labels = set()
    for prefixe, champ in (("genre", genres), ("tag", tags)):
        for valeur in champ.split("|"):
            label = " ".join(unicodedata.normalize("NFKC", valeur).casefold().split())
            if label:
                labels.add(f"{prefixe}:{label}")
    return frozenset(labels)


def creer_shingles_mots(texte, k, prefixe=""):
    """Shingles de k mots consécutifs (texte déjà nettoyé)."""
    mots = texte.split()
    return frozenset(prefixe + " ".join(mots[i:i + k]) for i in range(len(mots) - k + 1))


def signatures_vocabulaire_reduit(documents, fonctions):
    """Signatures MinHash en hachant chaque shingle une seule fois pour tout le corpus.

    Rapide quand le vocabulaire est petit (genres, tags) ; trop gourmand en mémoire
    pour les shingles de synopsis : utiliser alors `signatures_par_document`.
    """
    valeurs = {}
    for shingles in documents.values():
        for shingle in shingles:
            if shingle not in valeurs:
                identifiant = hacher_shingle(shingle)
                valeurs[shingle] = tuple((a * identifiant + b) % NEXT_PRIME for a, b in fonctions)

    cache = {}
    signatures = []
    for shingles in documents.values():
        if not shingles:
            raise ValueError("Un ensemble vide ne doit pas recevoir de signature MinHash.")
        if shingles not in cache:
            cache[shingles] = tuple(min(composantes) for composantes in zip(
                *(valeurs[shingle] for shingle in shingles)
            ))
        signatures.append(cache[shingles])
    return signatures


def signatures_par_document(documents, fonctions):
    """Signatures MinHash calculées document par document (mémoire constante)."""
    signatures = []
    for shingles in documents.values():
        if not shingles:
            raise ValueError("Un ensemble vide ne doit pas recevoir de signature MinHash.")
        identifiants = {hacher_shingle(shingle) for shingle in shingles}
        signatures.append(tuple(
            min((a * identifiant + b) % NEXT_PRIME for identifiant in identifiants)
            for a, b in fonctions
        ))
    return signatures


def exporter_similarites(documents, signatures, destination, seuil=0.5):
    """Écrire les paires dont le Jaccard estimé dépasse strictement `seuil`.

    Colonnes : article1, article2, Est. J, Act. J (Jaccard exact pour vérification).
    """
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


def ajouter_arguments_minhash(parser, entree, sortie, seuil):
    """Options communes aux méthodes MinHash."""
    parser.add_argument("--input", type=Path, default=entree)
    parser.add_argument("--output", type=Path, default=sortie)
    parser.add_argument("--num-hashes", type=int, default=NUM_HASHES)
    parser.add_argument("--seed", type=int, default=HASH_SEED)
    parser.add_argument("--threshold", type=float, default=seuil)


def verifier_arguments_minhash(args):
    if args.num_hashes < 1 or not 0 <= args.threshold <= 1:
        raise ValueError("--num-hashes doit être positif et --threshold compris entre 0 et 1.")


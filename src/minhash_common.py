"""Paramètres et fonctions de hachage communs aux traitements MinHash."""

import random


NUM_HASHES = 128
HASH_SEED = 42
NEXT_PRIME = 4294967311  # Premier supérieur aux identifiants SHA-1 sur 32 bits.


def creer_fonctions_hachage(nombre=NUM_HASHES, seed=HASH_SEED):
    if nombre < 1:
        raise ValueError("Le nombre de fonctions de hachage doit être positif.")
    rng = random.Random(seed)
    coefficients_a = rng.sample(range(1, NEXT_PRIME), nombre)
    return [(a, rng.randrange(NEXT_PRIME)) for a in coefficients_a]

"""Item-KNN : similarité cosinus entre animes, calculée sur les listes des utilisateurs.

sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i|) * sqrt(|U_j|) + shrink)
On ne garde que les `k` voisins les plus proches de chaque anime ; le score d'un
anime pour un utilisateur est la somme de ses similarités avec les animes vus.
"""

import time

import numpy as np

from common.donnees import charger_train
from common.evaluation import ajouter_arguments_evaluation, lancer_evaluation

DESCRIPTION = "Item-KNN cosinus sur les co-visionnages (filtrage collaboratif)"


def entrainer(X, k=100, shrink=10.0):
    X = (X > 0).astype(np.float32)
    cooc = (X.T @ X).toarray()
    normes = np.sqrt(np.diag(cooc))
    S = cooc / (np.outer(normes, normes) + shrink)
    np.fill_diagonal(S, 0.0)
    if k < S.shape[0]:
        # Garder les k plus fortes similarités de chaque colonne.
        seuils = -np.partition(-S, k - 1, axis=0)[k - 1]
        S[S < seuils] = 0.0
    return S


def ajouter_arguments(parser):
    parser.add_argument("--k", type=int, default=100, help="voisins gardés par anime")
    parser.add_argument("--shrink", type=float, default=10.0,
                        help="pénalise les similarités calculées sur peu d'utilisateurs")
    ajouter_arguments_evaluation(parser)


def executer(args):
    debut = time.perf_counter()
    S = entrainer(charger_train(), args.k, args.shrink)
    print(f"Matrice de similarité {S.shape} calculée en {time.perf_counter() - debut:.1f} s.")
    lancer_evaluation("item_knn", lambda X: X @ S, args, {"k": args.k, "shrink": args.shrink})

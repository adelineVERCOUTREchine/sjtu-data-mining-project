"""EASE^R (Steck, 2019) : modèle linéaire item-item en forme fermée.

On cherche B (animes x animes) minimisant ||X - XB||² + λ||B||² avec diag(B) = 0 :
    P = (XᵀX + λI)⁻¹,   B = I - P · diag(1 / diag(P))
Le score d'un utilisateur est x_u · B : un nouveau profil MAL est intégré sans
réentraînement, ce qui convient à la démo.
"""

import time

import numpy as np

import config
from common.donnees import charger_train
from common.evaluation import ajouter_arguments_evaluation, lancer_evaluation

DESCRIPTION = "EASE^R : régression linéaire item-item en forme fermée"


def entrainer(X, lam=500.0):
    X = (X > 0).astype(np.float64)
    G = (X.T @ X).toarray()
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)


def ajouter_arguments(parser):
    parser.add_argument("--lam", type=float, default=500.0,
                        help="régularisation L2 (λ) ; à régler sur le split val")
    parser.add_argument("--save", action="store_true",
                        help="sauvegarder B dans models/ease.npy (pour la démo)")
    ajouter_arguments_evaluation(parser)


def executer(args):
    debut = time.perf_counter()
    B = entrainer(charger_train(), args.lam)
    print(f"Matrice B {B.shape} calculée en {time.perf_counter() - debut:.1f} s.")
    if args.save:
        np.save(config.MODELS_DIR / "ease.npy", B)
        print(f"B sauvegardée dans {config.MODELS_DIR / 'ease.npy'}")
    lancer_evaluation("ease", lambda X: X @ B, args, {"lam": args.lam})

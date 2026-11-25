"""Baseline popularité : recommande les animes les plus vus, identiques pour tous.

Toute méthode personnalisée doit faire mieux : c'est la référence du tableau comparatif.
"""

import numpy as np

from common.donnees import charger_train
from common.evaluation import ajouter_arguments_evaluation, lancer_evaluation

DESCRIPTION = "Baseline : animes les plus populaires (non personnalisé)"


def entrainer(X):
    return np.asarray((X > 0).sum(axis=0), dtype=np.float32).ravel()


def ajouter_arguments(parser):
    ajouter_arguments_evaluation(parser)


def executer(args):
    popularite = entrainer(charger_train())

    def scorer(X):
        return np.broadcast_to(popularite, X.shape).copy()

    lancer_evaluation("popularity", scorer, args, {})

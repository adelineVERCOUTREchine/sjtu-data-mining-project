"""Évaluation hors ligne commune à toutes les méthodes de recommandation.

Une méthode fournit `scorer(X)` : matrice creuse (utilisateurs x animes) des animes
connus -> scores denses de même forme. Le protocole est identique pour toutes :
animes déjà vus exclus, puis top-K comparé aux animes masqués.
"""

import csv
import json
import time
from datetime import datetime

import numpy as np

import config
from common.donnees import charger_items, charger_split, charger_train

K_LIST = (10, 20, 50)
DOSSIER = config.RESULTS_DIR / "recommandation"


def ajouter_arguments_evaluation(parser):
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="val pour régler les paramètres, test pour le chiffre final")
    parser.add_argument("--exemple", type=int, default=16498,
                        help="mal_id d'un anime pour afficher ses voisins (0 = aucun)")
    parser.add_argument("--batch-size", type=int, default=2000)


def top_k(scores, k):
    """Indices des k meilleurs scores par ligne, triés par score décroissant."""
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1, kind="stable")
    partiel = np.argpartition(-scores, k, axis=1)[:, :k]
    ordre = np.argsort(-np.take_along_axis(scores, partiel, axis=1), axis=1)
    return np.take_along_axis(partiel, ordre, axis=1)


def evaluer(scorer, split, popularite, batch_size=2000):
    """Recall@K, nDCG@K, HitRate@K, couverture et popularité moyenne des top-10."""
    k_max = max(K_LIST)
    n_users, n_items = split.input.shape
    remise = 1 / np.log2(np.arange(2, k_max + 2))
    sommes = {f"{m}@{k}": 0.0 for k in K_LIST for m in ("recall", "ndcg", "hitrate")}
    recommandes = np.zeros(n_items, dtype=bool)
    pop_moyenne = 0.0
    evalues = 0

    for debut in range(0, n_users, batch_size):
        fin = min(debut + batch_size, n_users)
        cibles = split.target[debut:fin]
        scores = np.asarray(scorer(split.input[debut:fin]), dtype=np.float32)
        exclus = split.exclude[debut:fin].tocoo()
        scores[exclus.row, exclus.col] = -np.inf
        meilleurs = top_k(scores, k_max)
        touches = np.asarray(cibles[np.arange(fin - debut)[:, None], meilleurs].todense()) > 0
        n_cibles = np.diff(cibles.indptr)
        valides = n_cibles > 0
        for k in K_LIST:
            hits = touches[:, :k]
            dcg = (hits * remise[:hits.shape[1]]).sum(axis=1)  # liste < k si petit catalogue
            idcg = np.array([remise[:min(n, k)].sum() for n in n_cibles])
            sommes[f"recall@{k}"] += (hits.sum(axis=1)[valides] / np.minimum(n_cibles, k)[valides]).sum()
            sommes[f"ndcg@{k}"] += (dcg[valides] / idcg[valides]).sum()
            sommes[f"hitrate@{k}"] += hits.any(axis=1)[valides].sum()
        recommandes[meilleurs[:, :10].ravel()] = True
        pop_moyenne += popularite[meilleurs[valides, :10]].mean(axis=1).sum()
        evalues += valides.sum()

    resultats = {nom: valeur / evalues for nom, valeur in sommes.items()}
    resultats["coverage@10"] = recommandes.mean()
    resultats["popularite_moyenne@10"] = pop_moyenne / evalues
    resultats["utilisateurs"] = int(evalues)
    return resultats


def afficher_voisins(scorer, mal_id, n=10):
    """Animes recommandés à partir d'un seul anime (vecteur indicateur)."""
    import scipy.sparse as sp
    items = charger_items()
    ligne = items.index[items.mal_id == mal_id]
    if len(ligne) == 0:
        print(f"mal_id {mal_id} absent du catalogue des interactions.")
        return
    i = int(ligne[0])
    x = sp.csr_matrix(([1.0], ([0], [i])), shape=(1, len(items)))
    scores = np.asarray(scorer(x), dtype=np.float32)
    scores[0, i] = -np.inf
    print(f"\nRecommandations à partir de « {items.titre[i]} » :")
    for rang, j in enumerate(top_k(scores, n)[0], 1):
        print(f"  {rang:>2}. {items.titre[j]}  (mal_id {items.mal_id[j]}, score {scores[0, j]:.3f})")


def lancer_evaluation(nom, scorer, args, params):
    """Évaluer, afficher, sauvegarder dans results/recommandation/."""
    train = charger_train()
    popularite = np.asarray((train > 0).mean(axis=0)).ravel()
    split = charger_split(args.split)

    debut = time.perf_counter()
    resultats = evaluer(scorer, split, popularite, args.batch_size)
    duree = time.perf_counter() - debut

    print(f"\n{nom} — split {args.split} ({resultats['utilisateurs']} utilisateurs, {duree:.1f} s)")
    for cle, valeur in resultats.items():
        if cle != "utilisateurs":
            print(f"  {cle:<24} {valeur:.4f}")

    DOSSIER.mkdir(parents=True, exist_ok=True)
    meta = config.PROCESSED_DIR / "meta.json"
    if meta.exists():  # dataset d'interactions utilisé (2020 ou 2025)
        params = {**params, "donnees": json.loads(meta.read_text(encoding="utf-8"))
                  ["params"].get("source", "2020")}
    ligne = {"date": datetime.now().isoformat(timespec="seconds"), "methode": nom,
             "split": args.split, "params": json.dumps(params, sort_keys=True), **resultats}
    tableau = DOSSIER / "resultats.csv"
    nouveau = not tableau.exists()
    with tableau.open("a", encoding="utf-8", newline="") as sortie:
        ecrivain = csv.DictWriter(sortie, fieldnames=list(ligne))
        if nouveau:
            ecrivain.writeheader()
        ecrivain.writerow(ligne)
    print(f"Résultats ajoutés à {tableau}")

    if args.exemple:
        afficher_voisins(scorer, args.exemple)
    return resultats

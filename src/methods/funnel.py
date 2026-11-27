"""Recommandation en entonnoir : retrieval multi-sources puis re-ranking LightGBM.

Étape 1 (retrieval) : chaque générateur propose ses meilleurs animés non vus ;
l'union forme les candidats (~250 par utilisateur) :
    EASE (co-visionnages), Item-KNN (cosinus), embeddings de synopsis (contenu),
    popularité.
Étape 2 (re-ranking) : un modèle LambdaMART (LightGBM, objectif lambdarank)
classe les candidats à partir de caractéristiques :
    scores et rangs de chaque générateur, popularité, note MAL, année, format,
    nombre d'épisodes, affinité de genres avec le profil, similarité de synopsis
    maximale avec un animé vu, appartenance à une franchise déjà vue, taille du profil.

Éviter la fuite : le re-ranker apprend sur des utilisateurs d'entraînement que les
générateurs n'ont PAS vus (--ranker-users, mis de côté avant d'entraîner EASE,
Item-KNN...). Sinon les scores EASE seraient gonflés sur leurs animés masqués et le
re-ranker apprendrait à lui faire une confiance aveugle.

Sorties : deux lignes dans results/recommandation/resultats.csv
    « ease (funnel, base) »  EASE seul, entraîné sur les mêmes utilisateurs
    « funnel »               entonnoir complet
et le rappel de l'étape 1 (part des animés masqués présents parmi les candidats).
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config
from common.donnees import charger_items, charger_split, charger_train
from common.evaluation import ajouter_arguments_evaluation, lancer_evaluation
from methods import ease, item_knn
from methods.synopsis_embeddings import calculer_embeddings, charger_textes
from methods.synopsis_finetune import matrice_franchise

DESCRIPTION = "Entonnoir : retrieval (EASE, KNN, contenu, popularité) + re-ranking LightGBM"
GENERATEURS = ("ease", "knn", "contenu", "popularite")
CARACTERISTIQUES = [
    "ease_score", "ease_rang", "knn_score", "knn_rang", "contenu_score", "contenu_rang",
    "contenu_max", "popularite_log", "pop_rang", "nb_generateurs", "note_mal", "annee",
    "ecart_annee", "episodes_log", "format", "affinite_genres", "franchise_vue",
    "taille_profil_log",
]


class Entonnoir:
    """Générateurs de candidats + caractéristiques pour un lot d'utilisateurs."""

    def __init__(self, X_train, args):
        debut = time.perf_counter()
        self.n = X_train.shape[1]
        self.par_generateur = args.candidates_per_retriever
        self.B = ease.entrainer(X_train, args.lam)
        self.S_knn = item_knn.entrainer(X_train, args.knn_k, args.knn_shrink)
        self.popularite = np.asarray((X_train > 0).sum(axis=0), dtype=np.float32).ravel()

        items = charger_items()
        catalogue = charger_textes()
        position = pd.Series(np.arange(len(catalogue)), index=catalogue.mal_id)
        E = calculer_embeddings(catalogue, args.model, False)[
            position.loc[items.mal_id].to_numpy()]
        self.S_contenu = E @ E.T
        np.fill_diagonal(self.S_contenu, 0.0)
        self.franchise = matrice_franchise(items.mal_id.to_numpy()).astype(np.float32)
        np.fill_diagonal(self.franchise, 0.0)

        meta = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "score", "start_year",
                                                      "episodes", "format", "genres"])
        meta = items[["mal_id"]].merge(meta, on="mal_id", how="left")
        self.note = meta.score.fillna(meta.score.mean()).to_numpy(np.float32)
        self.annee = meta.start_year.fillna(meta.start_year.median()).to_numpy(np.float32)
        self.episodes = np.log1p(meta.episodes.fillna(1).to_numpy(np.float32))
        self.format = pd.Categorical(meta.format.fillna("?")).codes.astype(np.float32)
        genres = sorted({g for gs in meta.genres.fillna("") for g in gs.split("|") if g})
        index = {g: k for k, g in enumerate(genres)}
        G = np.zeros((self.n, len(genres)), dtype=np.float32)
        for i, gs in enumerate(meta.genres.fillna("")):
            for g in gs.split("|"):
                if g in index:
                    G[i, index[g]] = 1.0
        self.genres = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        print(f"Générateurs prêts en {time.perf_counter() - debut:.1f} s.", flush=True)

    @staticmethod
    def _rangs(scores):
        """Rang de chaque animé (0 = meilleur) dans chaque ligne."""
        ordre = np.argsort(-scores, axis=1)
        rangs = np.empty_like(ordre)
        np.put_along_axis(rangs, ordre, np.arange(scores.shape[1])[None, :], axis=1)
        return rangs

    def candidats(self, X):
        """Scores des générateurs, rangs et masque des candidats (animés vus exclus)."""
        X = (X > 0).astype(np.float32).tocsr()
        vus = X.toarray() > 0
        scores = {
            "ease": np.asarray(X @ self.B), "knn": np.asarray(X @ self.S_knn),
            "contenu": np.asarray(X @ self.S_contenu),
            "popularite": np.broadcast_to(self.popularite, X.shape).copy(),
        }
        rangs, masque = {}, np.zeros(X.shape, dtype=bool)
        for nom, s in scores.items():
            s[vus] = -np.inf
            rangs[nom] = self._rangs(s)
            masque |= rangs[nom] < self.par_generateur
        masque &= ~vus
        return X, scores, rangs, masque

    def caracteristiques(self, X, scores, rangs, masque):
        """Matrice (nb candidats, nb caractéristiques), lignes ordonnées par utilisateur."""
        u, c = np.nonzero(masque)
        taille = np.asarray(X.sum(axis=1)).ravel()
        profil_genres = np.asarray(X @ self.genres)
        profil_genres /= np.maximum(np.linalg.norm(profil_genres, axis=1, keepdims=True), 1e-9)
        annee_moy = np.asarray(X @ self.annee).ravel() / np.maximum(taille, 1)
        franchise = np.asarray(X @ self.franchise) > 0
        contenu_max = np.zeros(len(u), dtype=np.float32)
        bornes = np.searchsorted(u, np.arange(X.shape[0] + 1))  # u est trié par ligne
        for ligne in range(X.shape[0]):
            vus = X.indices[X.indptr[ligne]:X.indptr[ligne + 1]]
            a, b = bornes[ligne], bornes[ligne + 1]
            if len(vus) and b > a:
                contenu_max[a:b] = self.S_contenu[np.ix_(vus, c[a:b])].max(axis=0)
        nb = sum((rangs[g][u, c] < self.par_generateur) for g in GENERATEURS)
        colonnes = {
            "ease_score": scores["ease"][u, c], "ease_rang": rangs["ease"][u, c],
            "knn_score": scores["knn"][u, c], "knn_rang": rangs["knn"][u, c],
            "contenu_score": scores["contenu"][u, c] / np.maximum(taille[u], 1),
            "contenu_rang": rangs["contenu"][u, c], "contenu_max": contenu_max,
            "popularite_log": np.log1p(self.popularite[c]),
            "pop_rang": rangs["popularite"][u, c], "nb_generateurs": nb,
            "note_mal": self.note[c], "annee": self.annee[c],
            "ecart_annee": self.annee[c] - annee_moy[u], "episodes_log": self.episodes[c],
            "format": self.format[c],
            "affinite_genres": (profil_genres[u] * self.genres[c]).sum(axis=1),
            "franchise_vue": franchise[u, c].astype(np.float32),
            "taille_profil_log": np.log1p(taille[u]),
        }
        return u, c, np.column_stack([colonnes[k] for k in CARACTERISTIQUES]).astype(np.float32)


def separer_profils(X, part, rng):
    """Masquer `part` des animés de chaque utilisateur : (entrées, cibles)."""
    X = X.tocoo()
    garde = rng.random(X.nnz) >= part
    forme = X.shape
    entree = sp.csr_matrix((X.data[garde], (X.row[garde], X.col[garde])), shape=forme)
    cible = sp.csr_matrix((X.data[~garde], (X.row[~garde], X.col[~garde])), shape=forme)
    return entree, cible


def donnees_ranker(entonnoir, entree, cible, batch_size):
    blocs, labels, groupes = [], [], []
    for debut in range(0, entree.shape[0], batch_size):
        X, scores, rangs, masque = entonnoir.candidats(entree[debut:debut + batch_size])
        u, c, F = entonnoir.caracteristiques(X, scores, rangs, masque)
        y = np.asarray(cible[debut:debut + batch_size][u, c]).ravel() > 0
        # Ne garder que les utilisateurs ayant au moins un bon candidat (signal).
        bornes = np.searchsorted(u, np.arange(X.shape[0] + 1))  # u est trié par ligne
        for a, b in zip(bornes[:-1], bornes[1:]):
            if b > a and y[a:b].any():
                blocs.append(F[a:b]); labels.append(y[a:b]); groupes.append(b - a)
    return np.vstack(blocs), np.concatenate(labels).astype(int), np.array(groupes)


def ajouter_arguments(parser):
    parser.add_argument("--candidates-per-retriever", type=int, default=100,
                        help="candidats proposés par chaque générateur (étape 1)")
    parser.add_argument("--ranker-users", type=int, default=5000,
                        help="utilisateurs d'entraînement réservés au re-ranker")
    parser.add_argument("--lam", type=float, default=500.0, help="EASE : régularisation")
    parser.add_argument("--knn-k", type=int, default=100, help="Item-KNN : voisins")
    parser.add_argument("--knn-shrink", type=float, default=10.0)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5",
                        help="embeddings de synopsis (cache models/embeddings/)")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--seed", type=int, default=config.SEED)
    ajouter_arguments_evaluation(parser)


def executer(args):
    import lightgbm as lgb

    rng = np.random.default_rng(args.seed)
    train = (charger_train() > 0).astype(np.float32).tocsr()
    ordre = rng.permutation(train.shape[0])
    ranker_ids, generateur_ids = ordre[:args.ranker_users], ordre[args.ranker_users:]
    entonnoir = Entonnoir(train[generateur_ids], args)

    # Données du re-ranker : profils réservés, 20 % masqués comme pour l'évaluation.
    debut = time.perf_counter()
    entree, cible = separer_profils(train[ranker_ids], 0.2, rng)
    F, y, groupes = donnees_ranker(entonnoir, entree, cible, args.batch_size)
    coupure = int(len(groupes) * 0.9)
    n_app = groupes[:coupure].sum()
    print(f"Re-ranker : {len(groupes)} utilisateurs, {len(y):,} candidats, "
          f"{y.mean():.1%} positifs ({time.perf_counter() - debut:.0f} s).", flush=True)
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=args.seed, verbose=-1)
    ranker.fit(F[:n_app], y[:n_app], group=groupes[:coupure],
               eval_set=[(F[n_app:], y[n_app:])], eval_group=[groupes[coupure:]],
               eval_at=[10], feature_name=CARACTERISTIQUES,
               callbacks=[lgb.early_stopping(30, verbose=False)])
    print(f"LightGBM : {ranker.best_iteration_ or args.n_estimators} arbres, "
          f"nDCG@10 interne {ranker.best_score_['valid_0']['ndcg@10']:.4f}")
    importance = pd.Series(ranker.booster_.feature_importance("gain"), CARACTERISTIQUES)
    print("Importance (gain) :", ", ".join(
        f"{k} {v / importance.sum():.0%}" for k, v in importance.sort_values()[::-1].items()
        if v / importance.sum() >= 0.01))

    # Rappel de l'étape 1 sur la validation.
    split = charger_split(args.split)
    trouves = total = 0
    for debut in range(0, split.input.shape[0], args.batch_size):
        _, _, _, masque = entonnoir.candidats(split.input[debut:debut + args.batch_size])
        t = split.target[debut:debut + args.batch_size].toarray() > 0
        trouves += (t & masque).sum(); total += t.sum()
    print(f"Étape 1 : {masque.sum(axis=1).mean():.0f} candidats par utilisateur ; "
          f"{trouves / total:.1%} des animés masqués sont parmi eux.")

    def scorer_entonnoir(X):
        X, scores, rangs, masque = entonnoir.candidats(X)
        u, c, F_lot = entonnoir.caracteristiques(X, scores, rangs, masque)
        sortie = np.full(X.shape, -np.inf, dtype=np.float32)
        sortie[u, c] = ranker.predict(F_lot)
        return sortie

    params = {k: getattr(args, k) for k in ("candidates_per_retriever", "ranker_users",
                                            "lam", "knn_k", "model", "n_estimators")}
    lancer_evaluation("ease (funnel, base)", lambda X: X @ entonnoir.B, args,
                      {"lam": args.lam, "users": len(generateur_ids)})
    lancer_evaluation("funnel", scorer_entonnoir, args, params)

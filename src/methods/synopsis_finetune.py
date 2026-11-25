"""Fine-tuning du Transformer de synopsis sur les co-visionnages MAL (cold-start).

But : que deux animes aux synopsis proches dans l'espace appris soient des animes
regardés par les mêmes personnes, pour pouvoir recommander un anime que personne
n'a encore noté (sortie récente) à partir de son seul synopsis.

Protocole :
  1. 20 % des animes des listes MAL sont mis de côté (« animes froids ») : ni
     leurs synopsis ni leurs interactions ne servent à l'entraînement.
  2. Paires positives : pour chaque anime d'entraînement, ses voisins
     collaboratifs les plus proches (cosinus des co-visionnages, comme Item-KNN)
     parmi les animes d'entraînement, en retirant les paires de la même franchise
     (titres trop semblables) qui n'apprendraient rien.
  3. Apprentissage contrastif (CachedMultipleNegativesSymmetricRankingLoss) : chaque
     synopsis doit être plus proche de son voisin que des autres synopsis du lot.
     La version « cached » (GradCache) calcule le lot par petits morceaux : grands
     lots (meilleurs négatifs) sans dépasser la mémoire d'un GPU T4.
  4. Évaluation sur les seuls animes froids, modèle de base contre modèle affiné :
       - item-to-item : les 10 voisins par le texte retrouvent-ils les 10 voisins
         collaboratifs (hors franchise) ? -> Recall@10
       - user-to-item : recommander uniquement parmi les animes froids aux
         utilisateurs de validation -> nDCG@10, Recall@10 (même code que les
         autres méthodes) ; repères : hasard et EASE, qui lui a vu les
         interactions (borne haute).

Sorties : models/embeddings/<modèle>-anime-ft.npz (vecteurs du catalogue complet,
réutilisables avec --model <modèle>-anime-ft dans synopsis_embeddings et
embeddings_map), results/finetune/<modèle>.csv (tableau comparatif).
À lancer sur GPU : python scripts/kaggle_run.py synopsis_finetune
"""

import csv
import re
import time

import numpy as np
import pandas as pd

import config
from common.donnees import Split, charger_items, charger_split, charger_train
from common.evaluation import evaluer, top_k
from methods import ease, item_knn
from methods.synopsis_embeddings import calculer_embeddings, charger_textes

DESCRIPTION = "Fine-tuning contrastif du Transformer sur les co-visionnages (cold-start)"
MODELE_DEFAUT = "BAAI/bge-base-en-v1.5"
MOTS_VIDES = {"the", "a", "an", "of", "no", "to", "wa", "ga", "season", "part", "movie",
              "film", "ova", "ona", "special", "specials", "final", "2nd", "3rd", "4th",
              "5th", "ii", "iii", "iv", "tv", "series", "and", "in", "on", "x"}


# --- Franchises -------------------------------------------------------------------

def jetons_titre(*titres):
    jetons = set()
    for titre in titres:
        if isinstance(titre, str):
            jetons |= {m for m in re.findall(r"[a-z0-9]+", titre.lower())
                       if m not in MOTS_VIDES and not m.isdigit()}
    return jetons


def matrice_franchise(mal_ids, seuil=0.5):
    """Booléen (n, n) : True si les titres partagent assez de mots (même franchise)."""
    import scipy.sparse as sp
    meta = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "title_romaji", "title_english"])
    meta = meta.set_index("mal_id").loc[mal_ids]
    ensembles = [jetons_titre(r.title_romaji, r.title_english) for r in meta.itertuples()]
    vocab = {m: k for k, m in enumerate(sorted(set().union(*ensembles)))}
    lignes = [i for i, e in enumerate(ensembles) for _ in e]
    colonnes = [vocab[m] for e in ensembles for m in e]
    T = sp.csr_matrix((np.ones(len(lignes), np.float32), (lignes, colonnes)),
                      shape=(len(ensembles), len(vocab)))
    inter = (T @ T.T).toarray()
    tailles = np.array([len(e) for e in ensembles], dtype=np.float32)
    union = tailles[:, None] + tailles[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 0.0)
    franchise = jaccard >= seuil
    np.fill_diagonal(franchise, True)
    return franchise


# --- Données d'entraînement --------------------------------------------------------

def separer_animes(n_items, part_froide, seed):
    rng = np.random.default_rng(seed)
    froids = np.zeros(n_items, dtype=bool)
    froids[rng.choice(n_items, size=int(round(n_items * part_froide)), replace=False)] = True
    return froids


def creer_paires(sim_cf, froids, franchise, par_anime):
    """Paires (i, j) d'animes d'entraînement : j parmi les voisins collaboratifs de i."""
    S = sim_cf.copy()
    S[:, froids] = -np.inf          # jamais d'anime froid dans les paires
    S[franchise] = -np.inf          # ni de paire de la même franchise
    paires = []
    for i in np.flatnonzero(~froids):
        voisins = np.argsort(-S[i])[:par_anime]
        paires.extend((i, int(j)) for j in voisins if np.isfinite(S[i, j]) and S[i, j] > 0)
    return paires


def affiner(modele, textes, paires, args, dossier):
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    cuda = torch.cuda.is_available()
    if not cuda:
        print("Attention : pas de GPU, le fine-tuning sera très lent.", flush=True)
    model = SentenceTransformer(modele, device="cuda" if cuda else "cpu")
    model.max_seq_length = args.max_seq_length
    jeu = Dataset.from_dict({"anchor": [textes[i] for i, _ in paires],
                             "positive": [textes[j] for _, j in paires]})
    reglages = SentenceTransformerTrainingArguments(
        output_dir=str(dossier), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size, learning_rate=args.lr,
        warmup_ratio=0.1, fp16=cuda, seed=args.seed, save_strategy="no",
        logging_steps=50, report_to="none",
        # Un même anime revient dans plusieurs paires : éviter qu'il serve de
        # « négatif » à lui-même dans un lot.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    entraineur = SentenceTransformerTrainer(
        model=model, args=reglages, train_dataset=jeu,
        loss=losses.CachedMultipleNegativesSymmetricRankingLoss(
            model, mini_batch_size=args.mini_batch_size))
    entraineur.train()
    return model


# --- Évaluation sur les animes froids ----------------------------------------------

def recall_item_to_item(sim_texte, sim_cf, froids, franchise, k=10):
    """Part des k voisins collaboratifs (hors franchise) retrouvés par le texte."""
    rappels = []
    for i in np.flatnonzero(froids):
        cf = sim_cf[i].copy()
        txt = sim_texte[i].copy()
        cf[franchise[i]] = -np.inf
        txt[franchise[i]] = -np.inf
        verite = set(np.argsort(-cf)[:k])
        rappels.append(len(verite & set(np.argsort(-txt)[:k])) / k)
    return float(np.mean(rappels))


def evaluer_froid(nom, scorer_complet, split, froids, popularite, batch_size):
    """Recommander uniquement parmi les animes froids (colonnes `froids`)."""
    sous = Split(split.nom, split.input, split.target[:, froids].tocsr(),
                 split.exclude[:, froids].tocsr())
    resultats = evaluer(lambda X: scorer_complet(X)[:, froids], sous,
                        popularite[froids], batch_size)
    resultats["methode"] = nom
    return resultats


def ajouter_arguments(parser):
    parser.add_argument("--model", default=MODELE_DEFAUT, help="modèle de départ")
    parser.add_argument("--cold-fraction", type=float, default=0.2,
                        help="part des animes mis de côté pour l'évaluation cold-start")
    parser.add_argument("--pairs-per-item", type=int, default=5,
                        help="voisins collaboratifs utilisés comme paires positives")
    parser.add_argument("--franchise-threshold", type=float, default=0.5,
                        help="Jaccard des mots des titres au-delà duquel 2 animes = franchise")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=64,
                        help="paires par lot (négatifs contrastifs = autres paires du lot)")
    parser.add_argument("--mini-batch-size", type=int, default=8,
                        help="morceaux calculés à la fois sur le GPU (mémoire)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=320)
    parser.add_argument("--save-model", action="store_true",
                        help="sauvegarder le modèle affiné (~440 Mo) dans models/")
    parser.add_argument("--batch-size", type=int, default=2000, help="évaluation")
    parser.add_argument("--seed", type=int, default=config.SEED)


def executer(args):
    debut = time.perf_counter()
    items = charger_items()
    train = charger_train()
    split = charger_split("val")
    n_items = len(items)

    catalogue = charger_textes()
    position = pd.Series(np.arange(len(catalogue)), index=catalogue.mal_id)
    lignes = position.loc[items.mal_id].to_numpy()       # colonnes -> lignes du catalogue
    textes = catalogue.texte.to_numpy()[lignes]

    froids = separer_animes(n_items, args.cold_fraction, args.seed)
    franchise = matrice_franchise(items.mal_id.to_numpy(), args.franchise_threshold)
    sim_cf = item_knn.entrainer(train, k=n_items)        # cosinus des co-visionnages
    paires = creer_paires(sim_cf, froids, franchise, args.pairs_per_item)
    print(f"{n_items} animes : {(~froids).sum()} d'entraînement, {froids.sum()} froids ; "
          f"{len(paires)} paires positives ; "
          f"{int(franchise.sum() - n_items) // 2} paires de franchise exclues.", flush=True)

    nom_base = args.model.split("/")[-1]
    dossier = config.MODELS_DIR / "finetune" / nom_base
    model = affiner(args.model, textes, paires, args, dossier)
    print(f"Fine-tuning terminé en {time.perf_counter() - debut:.0f} s.", flush=True)

    # Vecteurs du catalogue complet : base (cache) et affiné.
    base = calculer_embeddings(catalogue, args.model, False)
    affine = model.encode(catalogue.texte.tolist(), batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    cache = config.MODELS_DIR / "embeddings" / f"{nom_base}-anime-ft.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mal_id=catalogue.mal_id.to_numpy(), vecteurs=affine)
    print(f"Vecteurs affinés : {cache}")
    if args.save_model:
        model.save(str(dossier / "modele"))

    popularite = np.asarray((train > 0).mean(axis=0)).ravel()
    B = ease.entrainer(train)
    rng = np.random.default_rng(args.seed)
    lignes_res = [evaluer_froid("hasard", lambda X: rng.random(X.shape, dtype=np.float32),
                                split, froids, popularite, args.batch_size)]
    for nom, vecteurs in ((nom_base, base), (f"{nom_base} affiné", affine)):
        E = vecteurs[lignes]
        S = E @ E.T
        np.fill_diagonal(S, 0.0)
        res = evaluer_froid(nom, lambda X, S=S: X @ S, split, froids, popularite,
                            args.batch_size)
        res["item_recall@10"] = recall_item_to_item(S, sim_cf, froids, franchise)
        lignes_res.append(res)
    lignes_res.append(evaluer_froid("ease (borne haute, a vu les interactions)",
                                    lambda X: X @ B, split, froids, popularite,
                                    args.batch_size))

    colonnes = ["methode", "item_recall@10", "ndcg@10", "recall@10", "hitrate@10",
                "coverage@10", "popularite_moyenne@10", "utilisateurs"]
    print(f"\nAnimes froids ({froids.sum()}), utilisateurs de validation :")
    print(f"  {'méthode':<44}{'item R@10':>10}{'nDCG@10':>9}{'R@10':>8}{'Hit@10':>8}")
    for r in lignes_res:
        item = r.get("item_recall@10")
        print(f"  {r['methode']:<44}{'' if item is None else f'{item:.3f}':>10}"
              f"{r['ndcg@10']:>9.3f}{r['recall@10']:>8.3f}{r['hitrate@10']:>8.3f}")

    sortie = config.RESULTS_DIR / "finetune" / f"{nom_base}.csv"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes + ["params"],
                                  extrasaction="ignore")
        ecrivain.writeheader()
        params = {k: v for k, v in vars(args).items() if k not in ("batch_size",)}
        for r in lignes_res:
            ecrivain.writerow({**r, "params": params})
    print(f"Tableau : {sortie} ({time.perf_counter() - debut:.0f} s au total)")

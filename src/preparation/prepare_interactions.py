"""Construire les matrices utilisateurs x animes et les splits d'évaluation.

Entrée : data/raw/anime-recommendation-database-2020/animelist.csv (109 M lignes).
Sorties (data/processed/) :
  items.csv                 index de colonne -> mal_id, titre
  train.npz                 utilisateurs d'entraînement x animes, 1 = interaction positive
  train_ratings.npz         mêmes positions, note MAL (0 = non notée)
  {val,test}_input.npz      animes positifs visibles des utilisateurs de validation / test
  {val,test}_target.npz     20 % de leurs positifs, masqués : ce que les modèles doivent retrouver
  {val,test}_exclude.npz    tout ce qu'ils ont déjà vu hors cibles (à ne pas recommander)
  meta.json                 paramètres et statistiques

Définitions (statuts MAL : 1 en cours, 2 terminé, 3 en pause, 4 abandonné, 6 prévu) :
  positif = statut 1, 2 ou 3, sauf note de 1 à 4
  vu      = statut 1 à 4 ; « prévu » (6) est ignoré
Les utilisateurs de validation et de test sont absents de l'entraînement, comme un
profil MAL saisi dans la démo : les modèles doivent les intégrer sans réentraînement.
"""

import json
import time

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

import config

DESCRIPTION = "Filtrer animelist.csv, échantillonner les utilisateurs, créer les splits"


def ajouter_arguments(parser):
    parser.add_argument("--num-users", type=int, default=100_000,
                        help="nombre d'utilisateurs échantillonnés (0 = tous)")
    parser.add_argument("--min-positives", type=int, default=10,
                        help="nombre minimum d'animes positifs par utilisateur")
    parser.add_argument("--min-item-users", type=int, default=5,
                        help="nombre minimum d'utilisateurs positifs par anime")
    parser.add_argument("--val-users", type=int, default=5_000)
    parser.add_argument("--test-users", type=int, default=5_000)
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="part des positifs masqués pour les utilisateurs val/test")
    parser.add_argument("--seed", type=int, default=config.SEED)


def charger_interactions(args):
    source = config.INTERACTIONS_RAW_DIR / "animelist.csv"
    if not source.exists():
        raise FileNotFoundError(f"{source} absent : lancer d'abord download_interactions.")
    con = duckdb.connect()
    adultes = "WHERE NOT is_adult" if config.EXCLURE_ADULTES else ""
    con.execute(f"""
        CREATE TEMP TABLE catalogue AS
        SELECT DISTINCT CAST(mal_id AS INTEGER) AS anime_id
        FROM read_csv_auto('{config.ANIME_CSV.as_posix()}') {adultes}
    """)
    con.execute(f"""
        CREATE TEMP TABLE vues AS
        SELECT user_id::INTEGER AS user_id, anime_id::INTEGER AS anime_id,
               rating::TINYINT AS rating, watching_status::TINYINT AS statut,
               (watching_status IN (1, 2, 3) AND NOT rating BETWEEN 1 AND 4) AS positif
        FROM read_csv_auto('{source.as_posix()}')
        WHERE watching_status IN (1, 2, 3, 4)
          AND anime_id IN (SELECT anime_id FROM catalogue)
    """)
    limite = f"LIMIT {args.num_users}" if args.num_users else ""
    con.execute(f"""
        CREATE TEMP TABLE utilisateurs AS
        SELECT user_id FROM vues GROUP BY user_id
        HAVING sum(positif::INTEGER) >= {args.min_positives}
        ORDER BY hash(user_id + {args.seed} * 1000003), user_id {limite}
    """)
    df = con.execute("""
        SELECT v.* FROM vues v JOIN utilisateurs u USING (user_id)
    """).df()
    con.close()
    # Quelques couples (utilisateur, anime) apparaissent plusieurs fois.
    # DuckDB lit en parallèle : l'ordre des lignes varie d'une exécution (ou d'une
    # machine) à l'autre. Trier rend les splits identiques partout pour une même graine.
    df = df.sort_values(["user_id", "anime_id", "statut", "rating"], kind="stable")
    return df.drop_duplicates(["user_id", "anime_id"], keep="last").reset_index(drop=True)


def construire_matrice(lignes, colonnes, valeurs, forme):
    return sp.csr_matrix((valeurs, (lignes, colonnes)), shape=forme, dtype=np.float32)


def executer(args):
    debut = time.perf_counter()
    print("Lecture et filtrage d'animelist.csv avec DuckDB...", flush=True)
    df = charger_interactions(args)
    print(f"{len(df):,} interactions, {df.user_id.nunique():,} utilisateurs.", flush=True)

    rng = np.random.default_rng(args.seed)
    utilisateurs = rng.permutation(df.user_id.unique())
    n_eval = args.val_users + args.test_users
    if n_eval >= len(utilisateurs):
        raise ValueError("Pas assez d'utilisateurs pour l'entraînement.")
    role = pd.Series("train", index=utilisateurs)
    role.iloc[:args.val_users] = "val"
    role.iloc[args.val_users:n_eval] = "test"
    df["role"] = df.user_id.map(role)

    # Catalogue : animes suffisamment présents chez les utilisateurs d'entraînement.
    comptes = df[(df.role == "train") & df.positif].anime_id.value_counts()
    animes = np.sort(comptes[comptes >= args.min_item_users].index.to_numpy())
    df = df[df.anime_id.isin(animes)].copy()
    df["item"] = np.searchsorted(animes, df.anime_id.to_numpy())

    titres = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "title_english", "title_romaji"])
    titres["titre"] = titres.title_english.fillna(titres.title_romaji)
    items = pd.DataFrame({"index": np.arange(len(animes)), "mal_id": animes})
    items = items.merge(titres[["mal_id", "titre"]], on="mal_id", how="left")
    items.to_csv(config.PROCESSED_DIR / "items.csv", index=False)

    n_items = len(animes)
    stats = {"items": n_items}

    train = df[(df.role == "train") & df.positif]
    index_train = {u: i for i, u in enumerate(np.sort(train.user_id.unique()))}
    lignes = train.user_id.map(index_train).to_numpy()
    forme = (len(index_train), n_items)
    sp.save_npz(config.PROCESSED_DIR / "train.npz",
                construire_matrice(lignes, train.item, np.ones(len(train)), forme))
    sp.save_npz(config.PROCESSED_DIR / "train_ratings.npz",
                construire_matrice(lignes, train.item, train.rating.astype(np.float32), forme))
    stats["train"] = {"users": forme[0], "positives": len(train),
                      "density": len(train) / (forme[0] * n_items)}

    for nom in ("val", "test"):
        part = df[df.role == nom].reset_index(drop=True)
        index_users = {u: i for i, u in enumerate(np.sort(part.user_id.unique()))}
        part["ligne"] = part.user_id.map(index_users)
        positifs = part[part.positif]
        # Masquer une part des positifs de chaque utilisateur (au moins un).
        alea = rng.random(len(positifs))
        rang = positifs.assign(alea=alea).groupby("ligne").alea.rank(method="first")
        taille = positifs.groupby("ligne").ligne.transform("size")
        n_cibles = np.maximum(1, np.round(taille * args.holdout)).astype(int)
        est_cible = (rang <= n_cibles).reindex(part.index, fill_value=False)

        cibles = part[est_cible]
        entrees = part[part.positif & ~est_cible]
        exclus = part[~est_cible]
        forme = (len(index_users), n_items)
        for suffixe, bloc in (("input", entrees), ("target", cibles), ("exclude", exclus)):
            matrice = construire_matrice(bloc.ligne, bloc.item, np.ones(len(bloc)), forme)
            sp.save_npz(config.PROCESSED_DIR / f"{nom}_{suffixe}.npz", matrice)
        stats[nom] = {"users": forme[0], "input": len(entrees), "target": len(cibles)}

    meta = {"params": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "stats": stats}
    (config.PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                                     encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"Fichiers écrits dans {config.PROCESSED_DIR} ({time.perf_counter() - debut:.1f} s).")

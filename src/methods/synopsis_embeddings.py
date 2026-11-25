"""Embeddings de synopsis (Sentence Transformers) et similarité cosinus.

Chaque synopsis est encodé par un modèle de phrases pré-entraîné en un vecteur
normalisé ; deux animes sont proches si leurs vecteurs ont un cosinus élevé.
Contrairement à MinHash (shingles identiques), le modèle rapproche des thèmes
exprimés avec d'autres mots.

Couvre les 4 985 animes du catalogue, y compris les plus récents absents des
listes MAL de 2020 (cold-start). Sorties :
  models/embeddings/<modèle>.npz              vecteurs (cache, réutilisé ensuite)
  results/embeddings/voisins_<modèle>.csv     10 plus proches voisins de chaque anime
Évaluation user-to-item : le score d'un anime est la somme de ses similarités
avec les animes de l'utilisateur (comme Item-KNN, mais sans aucune interaction).
"""

import csv
import time

import numpy as np
import pandas as pd

import config
from common.donnees import charger_items, charger_train, masque_tout_public
from common.evaluation import ajouter_arguments_evaluation, lancer_evaluation, top_k
from preparation.prepare_content import retirer_credits

DESCRIPTION = "Embeddings de synopsis (Sentence Transformers) + cosinus (contenu)"
MODELE_DEFAUT = "sentence-transformers/all-MiniLM-L6-v2"


def charger_textes(avec_tags=False):
    """Catalogue complet : mal_id, titre et texte à encoder."""
    df = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "title_english", "title_romaji",
                                                "genres", "tags", "description"])
    df["titre"] = df.title_english.fillna(df.title_romaji)
    synopsis = df.description.fillna("").map(retirer_credits)
    # Sans synopsis, encoder au moins le titre.
    texte = synopsis.where(synopsis.str.split().str.len() >= 3, df.titre)
    if avec_tags:
        etiquettes = (df.genres.fillna("") + "|" + df.tags.fillna("")).map(
            lambda s: ", ".join(x for x in s.split("|") if x))
        texte = "Genres and themes: " + etiquettes + ". " + texte
    df["texte"] = texte
    return df[["mal_id", "titre", "texte"]].reset_index(drop=True)


def chemin_cache(modele, avec_tags):
    nom = modele.split("/")[-1] + ("_tags" if avec_tags else "")
    return config.MODELS_DIR / "embeddings" / f"{nom}.npz", nom


def calculer_embeddings(catalogue, modele, avec_tags, device=None, batch_size=64,
                        recalculer=False):
    """Vecteurs normalisés (n_animes, d), dans l'ordre de `catalogue`. Mis en cache."""
    cache, _ = chemin_cache(modele, avec_tags)
    if cache.exists() and not recalculer:
        donnees = np.load(cache)
        if np.array_equal(donnees["mal_id"], catalogue.mal_id.to_numpy()):
            print(f"Embeddings chargés depuis {cache}")
            return donnees["vecteurs"]

    from sentence_transformers import SentenceTransformer
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encodage de {len(catalogue)} textes avec {modele} ({device})...", flush=True)
    debut = time.perf_counter()
    encodeur = SentenceTransformer(modele, device=device)
    vecteurs = encodeur.encode(catalogue.texte.tolist(), batch_size=batch_size,
                               normalize_embeddings=True, show_progress_bar=True,
                               convert_to_numpy=True).astype(np.float32)
    print(f"Encodage : {time.perf_counter() - debut:.1f} s ; dimension {vecteurs.shape[1]}; "
          f"longueur max {encodeur.max_seq_length} tokens (au-delà, texte tronqué).")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mal_id=catalogue.mal_id.to_numpy(), vecteurs=vecteurs)
    return vecteurs


def exporter_voisins(catalogue, vecteurs, destination, n=10):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as sortie:
        ecrivain = csv.writer(sortie)
        ecrivain.writerow(["mal_id", "titre", "rang", "voisin_mal_id", "voisin_titre", "cosinus"])
        for debut in range(0, len(vecteurs), 1000):
            sim = vecteurs[debut:debut + 1000] @ vecteurs.T
            np.fill_diagonal(sim[:, debut:debut + 1000], -np.inf)
            for ligne, voisins in enumerate(top_k(sim, n)):
                i = debut + ligne
                for rang, j in enumerate(voisins, 1):
                    ecrivain.writerow([catalogue.mal_id[i], catalogue.titre[i], rang,
                                       catalogue.mal_id[j], catalogue.titre[j],
                                       f"{sim[ligne, j]:.4f}"])


def afficher_voisins_contenu(catalogue, vecteurs, mal_id, n=10):
    lignes = catalogue.index[catalogue.mal_id == mal_id]
    if len(lignes) == 0:
        print(f"mal_id {mal_id} absent du catalogue.")
        return
    i = int(lignes[0])
    sim = vecteurs @ vecteurs[i]
    sim[i] = -np.inf
    print(f"\nAnimes au synopsis le plus proche de « {catalogue.titre[i]} » :")
    for rang, j in enumerate(np.argsort(-sim)[:n], 1):
        print(f"  {rang:>2}. {catalogue.titre[j]}  (mal_id {catalogue.mal_id[j]}, "
              f"cosinus {sim[j]:.3f})")


def matrice_similarite(catalogue, vecteurs, k=0):
    """Similarités entre les animes des matrices d'interactions (même ordre de colonnes)."""
    position = pd.Series(np.arange(len(catalogue)), index=catalogue.mal_id)
    E = vecteurs[position.loc[charger_items().mal_id].to_numpy()]
    S = E @ E.T
    np.fill_diagonal(S, 0.0)
    if 0 < k < S.shape[0]:
        seuils = -np.partition(-S, k - 1, axis=0)[k - 1]
        S[S < seuils] = 0.0
    return S


def ajouter_arguments(parser):
    parser.add_argument("--model", default=MODELE_DEFAUT,
                        help="modèle Sentence Transformers (Hugging Face)")
    parser.add_argument("--with-tags", action="store_true",
                        help="ajouter genres et tags au texte encodé")
    parser.add_argument("--k", type=int, default=0,
                        help="voisins gardés par anime pour l'évaluation (0 = tous)")
    parser.add_argument("--device", default=None, help="cpu ou cuda (défaut : auto)")
    parser.add_argument("--recompute", action="store_true", help="ignorer le cache")
    parser.add_argument("--no-eval", action="store_true",
                        help="calculer seulement les embeddings et les voisins")
    ajouter_arguments_evaluation(parser)


def executer(args):
    catalogue = charger_textes(args.with_tags)
    vecteurs = calculer_embeddings(catalogue, args.model, args.with_tags, args.device,
                                   recalculer=args.recompute)
    _, nom = chemin_cache(args.model, args.with_tags)
    # Le cache couvre tout le catalogue ; les animes adultes sont retirés ensuite.
    garder = masque_tout_public(catalogue.mal_id)
    catalogue, vecteurs = catalogue[garder].reset_index(drop=True), vecteurs[garder]
    voisins = config.RESULTS_DIR / "embeddings" / f"voisins_{nom}.csv"
    exporter_voisins(catalogue, vecteurs, voisins)
    print(f"10 plus proches voisins de chaque anime : {voisins}")
    if args.exemple:
        afficher_voisins_contenu(catalogue, vecteurs, args.exemple)
    if args.no_eval:
        return

    charger_train()  # vérifie que prepare_interactions a été lancé
    S = matrice_similarite(catalogue, vecteurs, args.k)
    args.exemple = 0  # voisins déjà affichés sur tout le catalogue
    lancer_evaluation(f"synopsis_embeddings[{nom}]", lambda X: X @ S, args,
                      {"model": args.model, "with_tags": args.with_tags, "k": args.k})

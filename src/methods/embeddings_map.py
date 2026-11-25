"""Carte 2D des animes : projection UMAP de leurs vecteurs (embeddings Transformer).

Deux sources de vecteurs :
  synopsis      embeddings de texte (Sentence Transformers, cache models/embeddings/)
  collaboratif  co-visionnages des listes MAL (SVD de la matrice utilisateurs x animes)

Par défaut, aucun regroupement : la carte montre seulement la projection. En option
(--clustering hdbscan ou kmeans), des groupes nommés par leurs étiquettes les plus
caractéristiques ; sur ce catalogue, HDBSCAN isole surtout les franchises.

La page HTML est autonome : survol, recherche d'un anime (surligné avec ses 10
plus proches voisins dans l'espace d'origine), mise en évidence d'un groupe ou
d'un genre, zoom et déplacement. Les animes adultes (is_adult) sont exclus.
Sortie : results/embeddings/carte_<source>.html
"""

import json
import time

import numpy as np
import pandas as pd

import config
from common.donnees import masque_tout_public
from common.evaluation import top_k
from methods.synopsis_embeddings import (
    MODELE_DEFAUT,
    calculer_embeddings,
    charger_textes,
    chemin_cache,
)

DESCRIPTION = "Carte UMAP interactive (HTML) des embeddings de synopsis"


def projeter(vecteurs, dimensions=2, n_neighbors=30, min_dist=0.0, seed=config.SEED):
    import umap
    reducteur = umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors,
                          min_dist=min_dist, metric="cosine", random_state=seed)
    return reducteur.fit_transform(vecteurs)


def vecteurs_collaboratifs(dimension=64, seed=config.SEED):
    """Vecteurs d'animes issus des co-visionnages : SVD de la matrice utilisateurs x animes.

    Les colonnes sont divisées par sqrt(popularité) pour que les animes très vus
    ne dominent pas tous les facteurs. Seuls les animes présents dans les listes
    MAL (data/processed/items.csv) sont placés sur la carte.
    """
    from sklearn.decomposition import TruncatedSVD
    from common.donnees import charger_items, charger_train
    X = (charger_train() > 0).astype(np.float32)
    popularite = np.asarray(X.sum(axis=0)).ravel()
    X = X.multiply(1 / np.sqrt(np.maximum(popularite, 1))).tocsr()
    svd = TruncatedSVD(n_components=dimension, random_state=seed)
    svd.fit(X)
    V = (svd.components_.T * svd.singular_values_).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    catalogue = charger_items()[["mal_id", "titre"]].copy()
    catalogue["titre"] = catalogue.titre.fillna(catalogue.mal_id.astype(str))
    return catalogue.reset_index(drop=True), V


def regrouper(vecteurs, args):
    """Étiquette de groupe par anime ; -1 = non classé ; None = pas de groupes."""
    if args.clustering == "none":
        return None
    if args.clustering == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=args.clusters, n_init=10,
                      random_state=args.seed).fit_predict(vecteurs)
    from sklearn.cluster import HDBSCAN
    reduits = projeter(vecteurs, 10, args.n_neighbors, 0.0, args.seed)
    # « leaf » garde les groupes les plus fins de la hiérarchie : évite qu'un seul
    # groupe géant absorbe presque tout le catalogue (cas de « eom »).
    return HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=10,
                   cluster_selection_method=args.cluster_selection).fit_predict(reduits)


def nommer_groupes(groupes, etiquettes, membres_mal, titres, n_mots=3, support_min=5):
    """Nom de chaque groupe : ses étiquettes les plus caractéristiques + un exemple.

    Score de l'étiquette l dans le groupe c : p(l|c) * log(p(l|c) / p(l)), qui
    favorise ce qui est fréquent dans le groupe et rare ailleurs. Deux groupes ne
    reçoivent jamais le même ensemble d'étiquettes (on prend alors les suivantes).
    """
    n = len(groupes)
    total = {}
    for labels in etiquettes:
        for label in set(labels):
            total[label] = total.get(label, 0) + 1
    noms, deja = [], set()
    for c in range(groupes.max() + 1):
        membres = np.flatnonzero(groupes == c)
        comptes = {}
        for i in membres:
            for label in set(etiquettes[i]):
                comptes[label] = comptes.get(label, 0) + 1
        scores = []
        for label, k in comptes.items():
            if k >= support_min:
                p_c = k / len(membres)
                scores.append((p_c * np.log(p_c / (total[label] / n)), label))
        classement = [label for _, label in sorted(scores, reverse=True)]
        choix = classement[:n_mots]
        # Éviter deux noms identiques, même dans un autre ordre.
        suivant = n_mots
        while frozenset(choix) in deja and suivant < len(classement):
            choix = classement[:n_mots - 1] + [classement[suivant]]
            suivant += 1
        deja.add(frozenset(choix))
        exemple = titres[membres[np.argmax(membres_mal[membres])]]
        noms.append((" · ".join(choix) or f"Groupe {c + 1}") + f" — ex. {exemple}")
    return noms


def coherence_genres(vecteurs, genres, n_voisins=10, seed=config.SEED):
    """Jaccard moyen des genres entre chaque anime et ses voisins, et entre paires au hasard.

    1 = voisins aux genres identiques ; la valeur « hasard » est le niveau obtenu
    sans aucune structure. L'écart mesure à quel point l'espace regroupe les genres.
    """
    ensembles = [frozenset(g) for g in genres]

    def jaccard(a, b):
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    valeurs = []
    for debut in range(0, len(vecteurs), 1000):
        sim = vecteurs[debut:debut + 1000] @ vecteurs.T
        np.fill_diagonal(sim[:, debut:debut + 1000], -np.inf)
        for ligne, voisins in enumerate(top_k(sim, n_voisins)):
            i = debut + ligne
            valeurs.append(np.mean([jaccard(ensembles[i], ensembles[j]) for j in voisins]))
    rng = np.random.default_rng(seed)
    paires = rng.integers(0, len(ensembles), size=(20000, 2))
    hasard = np.mean([jaccard(ensembles[i], ensembles[j]) for i, j in paires if i != j])
    return float(np.mean(valeurs)), float(hasard)


def preparer_donnees(catalogue, vecteurs, coords, groupes, n_voisins=10):
    meta = pd.read_csv(config.ANIME_CSV, usecols=["mal_id", "genres", "tags", "start_year",
                                                  "score", "members", "format"])
    meta = catalogue[["mal_id", "titre"]].merge(meta, on="mal_id", how="left")
    etiquettes = [[x for x in f"{g}|{t}".split("|") if x]
                  for g, t in zip(meta.genres.fillna(""), meta.tags.fillna(""))]
    if groupes is None:
        noms_groupes, bruit, groupes = [], -1, np.full(len(meta), -1)
    else:
        noms_groupes = nommer_groupes(groupes, etiquettes,
                                      meta.members.fillna(0).to_numpy(), meta.titre.tolist())
        bruit = len(noms_groupes)  # index du pseudo-groupe « non classés »
        noms_groupes.append("Non classés")
    voisins = []
    for debut in range(0, len(vecteurs), 1000):
        sim = vecteurs[debut:debut + 1000] @ vecteurs.T
        np.fill_diagonal(sim[:, debut:debut + 1000], -np.inf)
        voisins.extend(top_k(sim, n_voisins).tolist())
    genres = sorted({g for gs in meta.genres.fillna("") for g in gs.split("|") if g})
    index_genre = {g: i for i, g in enumerate(genres)}
    genres_animes = [[index_genre[g] for g in str(gs).split("|") if g in index_genre]
                     for gs in meta.genres.fillna("")]
    coherence, hasard = coherence_genres(vecteurs, genres_animes)

    def entier(valeur):
        return None if pd.isna(valeur) else int(valeur)

    animes = [{
        "id": int(r.mal_id), "t": r.titre,
        "x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3),
        "g": genres_animes[i],
        "a": entier(r.start_year), "s": None if pd.isna(r.score) else float(r.score),
        "m": entier(r.members), "f": r.format if isinstance(r.format, str) else "",
        "v": voisins[i], "c": int(groupes[i]) if groupes[i] >= 0 else bruit,

    } for i, r in enumerate(meta.itertuples())]
    return {"genres": genres, "groupes": noms_groupes, "bruit": bruit, "animes": animes,
            "coherence": round(coherence, 3), "hasard": round(hasard, 3)}


def ecrire_html(donnees, destination, titre_modele):
    resume = (f"Jaccard moyen des genres avec les 10 plus proches voisins : "
              f"{donnees['coherence']:.2f}, contre {donnees['hasard']:.2f} entre animes "
              f"pris au hasard")
    if donnees["bruit"] > 0:
        classes = sum(a["c"] != donnees["bruit"] for a in donnees["animes"])
        resume = f"{donnees['bruit']} groupes regroupant {classes} animes ; " + resume
    page = GABARIT.replace("__N__", f"{len(donnees['animes']):,}".replace(",", " "))
    page = page.replace("__RESUME__", resume).replace("__MODELE__", titre_modele).replace(
        "__DONNEES__", json.dumps(donnees, ensure_ascii=False, separators=(",", ":")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")


def ajouter_arguments(parser):
    parser.add_argument("--source", choices=("synopsis", "collaboratif"), default="synopsis",
                        help="synopsis : embeddings de texte ; collaboratif : co-visionnages MAL")
    parser.add_argument("--model", default=MODELE_DEFAUT,
                        help="modèle Sentence Transformers (cache réutilisé s'il existe)")
    parser.add_argument("--with-tags", action="store_true",
                        help="embeddings calculés avec genres et tags dans le texte")
    parser.add_argument("--clustering", choices=("none", "hdbscan", "kmeans"), default="none",
                        help="regroupement affiché sur la carte (défaut : aucun)")
    parser.add_argument("--clusters", type=int, default=16, help="KMeans : nombre de groupes")
    parser.add_argument("--min-cluster-size", type=int, default=40,
                        help="HDBSCAN : taille minimale d'un groupe")
    parser.add_argument("--cluster-selection", choices=("leaf", "eom"), default="leaf",
                        help="HDBSCAN : leaf = groupes fins, eom = groupes les plus stables")
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="UMAP : taille du voisinage (grand = structure globale)")
    parser.add_argument("--min-dist", type=float, default=0.0,
                        help="UMAP : écart minimal entre points (petit = groupes serrés)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=None, help="cpu ou cuda (défaut : auto)")


def executer(args):
    if args.source == "collaboratif":
        catalogue, vecteurs = vecteurs_collaboratifs(seed=args.seed)
        nom, titre = "collaboratif", "collaboratifs (SVD des listes MAL)"
    else:
        catalogue = charger_textes(args.with_tags)
        vecteurs = calculer_embeddings(catalogue, args.model, args.with_tags, args.device)
        _, nom = chemin_cache(args.model, args.with_tags)
        titre = f"de synopsis {nom}"

    garder = masque_tout_public(catalogue.mal_id)
    catalogue = catalogue[garder].reset_index(drop=True)
    vecteurs = vecteurs[garder]
    print(f"{(~garder).sum()} animes adultes exclus ; {len(catalogue)} animes sur la carte.")

    debut = time.perf_counter()
    groupes = regrouper(vecteurs, args)
    coords = projeter(vecteurs, 2, args.n_neighbors, args.min_dist, args.seed)
    print(f"Groupes et projection UMAP en {time.perf_counter() - debut:.1f} s.")

    donnees = preparer_donnees(catalogue, vecteurs, coords, groupes)
    for c, nom_groupe in enumerate(donnees["groupes"][:-1] if donnees["groupes"] else []):
        taille = sum(a["c"] == c for a in donnees["animes"])
        print(f"  groupe {c + 1:>2} ({taille:>4} animes) : {nom_groupe}")
    print(f"Cohérence des genres : voisins {donnees['coherence']:.3f}, "
          f"hasard {donnees['hasard']:.3f}")
    destination = config.RESULTS_DIR / "embeddings" / f"carte_{nom}.html"
    ecrire_html(donnees, destination, titre)
    print(f"Carte interactive : {destination}")


GABARIT = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Carte des animes</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f5f5f3; --surface-1: #fcfcfb; --border: #e2e1dc;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --context: #c9c8c2; --series-1: #2a78d6; --series-2: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #908f87;
    --context: #4a4a46; --series-1: #3987e5; --series-2: #d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #908f87;
  --context: #4a4a46; --series-1: #3987e5; --series-2: #d95926;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 16px 32px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sous-titre { color: var(--text-secondary); margin: 0 0 16px; max-width: 860px; }
.filtres { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 12px; }
.filtres label { display: flex; flex-direction: column; gap: 4px; font-size: 12px;
  color: var(--text-secondary); }
.filtres input, .filtres select, .filtres button { font: inherit; color: var(--text-primary);
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 10px; min-height: 34px; }
.filtres input { width: 280px; max-width: 100%; }
.filtres button { cursor: pointer; }
.mise-en-page { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; }
@media (max-width: 860px) { .mise-en-page { grid-template-columns: 1fr; } }
.carte { position: relative; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden; height: min(72vh, 760px); min-height: 360px; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
canvas.glisse { cursor: grabbing; }
.legende { display: flex; flex-wrap: wrap; gap: 16px; margin: 10px 2px 0;
  color: var(--text-secondary); font-size: 12px; }
.legende span { display: inline-flex; align-items: center; gap: 6px; }
.pastille { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.infobulle { position: absolute; pointer-events: none; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.12); max-width: 280px; display: none; }
.infobulle strong { display: block; font-size: 13px; margin-bottom: 2px; }
.infobulle .meta { color: var(--text-secondary); }
aside { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px; align-self: start; }
aside h2 { font-size: 14px; margin: 0 0 8px; }
aside p { color: var(--text-secondary); margin: 0; font-size: 13px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
td { padding: 5px 4px; border-top: 1px solid var(--border); vertical-align: top; }
td.num { text-align: right; color: var(--text-secondary); font-variant-numeric: tabular-nums;
  white-space: nowrap; }
td a { color: var(--text-primary); text-decoration: none; cursor: pointer; }
td a:hover { text-decoration: underline; }
.aide { color: var(--text-muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>Carte des animes</h1>
  <p class="sous-titre">Projection UMAP des vecteurs <b>__MODELE__</b> de __N__ animes.
  Chaque synopsis est encodé par un modèle Transformer en un vecteur ; deux points
  proches ont des synopsis proches. UMAP conserve les voisinages, pas les grandes
  distances. Les voisins listés sont calculés dans l'espace d'origine (cosinus).
  <br><b>__RESUME__</b>.</p>

  <div class="filtres">
    <label>Anime
      <input id="recherche" list="titres" placeholder="Rechercher un titre…" autocomplete="off">
      <datalist id="titres"></datalist>
    </label>
    <label>Mettre en évidence
      <select id="genre"><option value="">Aucun</option></select>
    </label>
    <label id="choix-noms" style="flex-direction:row;align-items:center;gap:6px;min-height:34px">
      <input type="checkbox" id="noms" checked style="width:auto;min-height:0"> Noms des groupes
    </label>
    <button id="reinit" type="button">Réinitialiser la vue</button>
  </div>

  <div class="mise-en-page">
    <div>
      <div class="carte" id="cadre">
        <canvas id="canvas" aria-label="Nuage de points des animes"></canvas>
        <div class="infobulle" id="infobulle"></div>
      </div>
      <div class="legende">
        <span><i class="pastille" style="background:var(--context)"></i>Autres animes</span>
        <span><i class="pastille" style="background:var(--series-1)"></i><b id="leg-genre">Genre</b>&nbsp;(choisir ci-dessus)</span>
        <span><i class="pastille" style="background:var(--series-2)"></i>Anime choisi et ses 10 voisins</span>
      </div>
      <div class="aide">Molette : zoom · glisser : déplacer · clic sur un point : le choisir</div>
    </div>
    <aside id="panneau">
      <h2>Voisins les plus proches</h2>
      <p>Choisissez un anime (recherche ou clic) pour voir ses 10 voisins.</p>
    </aside>
  </div>
</main>
<script>
const DONNEES = __DONNEES__;
const A = DONNEES.animes, G = DONNEES.genres, C = DONNEES.groupes;
const canvas = document.getElementById("canvas"), ctx = canvas.getContext("2d");
const cadre = document.getElementById("cadre"), bulle = document.getElementById("infobulle");
const parId = new Map(A.map((a, i) => [a.id, i]));
let choisi = -1, filtre = null, survol = -1, afficherNoms = true;
function enEvidence(a) { return filtre && (filtre[0] === "c" ? a.c === filtre[1] : a.g.includes(filtre[1])); }
let echelle = 1, decX = 0, decY = 0, largeur = 0, hauteur = 0;

const xs = A.map(a => a.x), ys = A.map(a => a.y);
const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
const nf = new Intl.NumberFormat("fr-FR");

function couleur(nom) { return getComputedStyle(document.documentElement).getPropertyValue(nom).trim(); }
function base(i) {
  const m = 24, l = largeur - 2 * m, h = hauteur - 2 * m;
  const k = Math.min(l / (maxX - minX), h / (maxY - minY));
  const ox = m + (l - k * (maxX - minX)) / 2, oy = m + (h - k * (maxY - minY)) / 2;
  return [ox + (A[i].x - minX) * k, oy + (maxY - A[i].y) * k];
}
function ecran(i) { const [x, y] = base(i); return [x * echelle + decX, y * echelle + decY]; }

function redimensionner() {
  const r = cadre.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  largeur = r.width; hauteur = r.height;
  canvas.width = largeur * dpr; canvas.height = hauteur * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  dessiner();
}

function point(i, rayon, fond, anneau) {
  const [x, y] = ecran(i);
  if (x < -10 || y < -10 || x > largeur + 10 || y > hauteur + 10) return;
  ctx.beginPath(); ctx.arc(x, y, rayon, 0, 2 * Math.PI);
  ctx.fillStyle = fond; ctx.fill();
  if (anneau) { ctx.lineWidth = 2; ctx.strokeStyle = anneau; ctx.stroke(); }
}

function dessiner() {
  const surface = couleur("--surface-1"), ctxCol = couleur("--context");
  const s1 = couleur("--series-1"), s2 = couleur("--series-2"), texte = couleur("--text-primary");
  ctx.fillStyle = surface; ctx.fillRect(0, 0, largeur, hauteur);
  const r = Math.max(2, Math.min(4, 2.2 * Math.sqrt(echelle)));
  const groupe = new Set(choisi >= 0 ? [choisi, ...A[choisi].v] : []);
  ctx.globalAlpha = filtre || choisi >= 0 ? 0.55 : 0.9;
  A.forEach((a, i) => { if (!groupe.has(i) && !enEvidence(a)) point(i, r, ctxCol); });
  ctx.globalAlpha = 1;
  if (filtre) A.forEach((a, i) => { if (!groupe.has(i) && enEvidence(a)) point(i, r + 0.5, s1, surface); });
  if (afficherNoms) nomsGroupes(texte, surface);
  if (choisi >= 0) {
    A[choisi].v.forEach(j => point(j, 5, s2, surface));
    point(choisi, 8, s2, surface);
    etiquette(choisi, texte, surface);
  }
  if (survol >= 0) { const [x, y] = ecran(survol);
    ctx.beginPath(); ctx.arc(x, y, r + 4, 0, 2 * Math.PI);
    ctx.lineWidth = 2; ctx.strokeStyle = texte; ctx.stroke(); }
}

// Nom de chaque groupe à la médiane de ses points ; un nom qui en chevauche un autre est omis.
function nomsGroupes(texte, surface) {
  const places = [];
  ctx.font = "600 12px system-ui, sans-serif";
  C.forEach((nom, c) => {
    if (c === DONNEES.bruit) return;
    const pts = A.map((a, i) => [a, i]).filter(([a]) => a.c === c).map(([, i]) => ecran(i));
    if (!pts.length) return;
    const med = k => pts.map(p => p[k]).sort((u, v) => u - v)[pts.length >> 1];
    nom = nom.split(" — ")[0];
    const x = med(0), y = med(1), l = ctx.measureText(nom).width;
    const boite = [x - l / 2 - 4, y - 9, x + l / 2 + 4, y + 9];
    if (places.some(b => !(boite[2] < b[0] || boite[0] > b[2] || boite[3] < b[1] || boite[1] > b[3]))) return;
    places.push(boite);
    ctx.lineWidth = 4; ctx.strokeStyle = surface; ctx.textAlign = "center";
    ctx.strokeText(nom, x, y + 4); ctx.fillStyle = texte; ctx.fillText(nom, x, y + 4);
    ctx.textAlign = "start";
  });
}

function etiquette(i, texte, surface) {
  const [x, y] = ecran(i);
  ctx.font = "600 12px system-ui, sans-serif";
  ctx.lineWidth = 4; ctx.strokeStyle = surface; ctx.strokeText(A[i].t, x + 11, y + 4);
  ctx.fillStyle = texte; ctx.fillText(A[i].t, x + 11, y + 4);
}

function plusProche(mx, my) {
  let meilleur = -1, d2min = 12 * 12;
  for (let i = 0; i < A.length; i++) {
    const [x, y] = ecran(i), d2 = (x - mx) ** 2 + (y - my) ** 2;
    if (d2 < d2min) { d2min = d2; meilleur = i; }
  }
  return meilleur;
}

function details(a) {
  const genres = a.g.map(k => G[k]).join(", ") || "—";
  const groupeNom = C.length && a.c >= 0 ? "Groupe : " + C[a.c] : "";
  const parts = [a.f, a.a, a.s != null ? "score " + a.s.toFixed(2) : null,
    a.m != null ? nf.format(a.m) + " membres" : null].filter(Boolean).join(" · ");
  return `<strong></strong><div class="meta"></div><div class="meta"></div>`
    .replace("<strong></strong>", "<strong>" + echapper(a.t) + "</strong>")
    .replace('<div class="meta"></div>', '<div class="meta">' + echapper(parts) + "</div>")
    .replace('<div class="meta"></div>', '<div class="meta">' + echapper(genres) + "</div>")
    + (groupeNom ? '<div class="meta">' + echapper(groupeNom) + "</div>" : "");
}
function echapper(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function choisir(i, centrer) {
  choisi = i;
  const panneau = document.getElementById("panneau");
  if (i < 0) { panneau.innerHTML = "<h2>Voisins les plus proches</h2><p>Choisissez un anime (recherche ou clic) pour voir ses 10 voisins.</p>"; dessiner(); return; }
  const a = A[i];
  const lignes = a.v.map((j, r) => `<tr><td class="num">${r + 1}</td><td><a data-i="${j}">${echapper(A[j].t)}</a></td><td class="num">${A[j].a ?? ""}</td></tr>`).join("");
  panneau.innerHTML = `<h2>${echapper(a.t)}</h2><p>${echapper([a.f, a.a].filter(Boolean).join(" · "))}</p>
    <table><tbody>${lignes}</tbody></table>
    <div class="aide"><a href="https://myanimelist.net/anime/${a.id}" target="_blank" rel="noopener" style="color:inherit">Voir sur MyAnimeList ↗</a></div>`;
  panneau.querySelectorAll("a[data-i]").forEach(el => el.addEventListener("click", () => choisir(+el.dataset.i, true)));
  if (centrer) { const [x, y] = base(i); echelle = Math.max(echelle, 2.5);
    decX = largeur / 2 - x * echelle; decY = hauteur / 2 - y * echelle; }
  dessiner();
}

// Filtres
const liste = document.getElementById("titres");
A.slice().sort((a, b) => a.t.localeCompare(b.t)).forEach(a => {
  const o = document.createElement("option"); o.value = a.t; liste.appendChild(o); });
document.getElementById("recherche").addEventListener("change", e => {
  const v = e.target.value.trim().toLowerCase();
  const i = A.findIndex(a => a.t.toLowerCase() === v);
  if (i >= 0) choisir(i, true);
});
const selGenre = document.getElementById("genre");
[["Groupes", C, "c", (k) => A.filter(a => a.c === k).length],
 ["Genres", G, "g", (k) => A.filter(a => a.g.includes(k)).length]].forEach(([titre, noms, type, compte]) => {
  if (!noms.length) return;
  const og = document.createElement("optgroup"); og.label = titre;
  noms.forEach((nom, k) => { const o = document.createElement("option"); o.value = type + ":" + k;
    o.textContent = `${nom} (${compte(k)})`; og.appendChild(o); });
  selGenre.appendChild(og); });
selGenre.addEventListener("change", e => {
  const [type, k] = e.target.value.split(":");
  filtre = type ? [type, +k] : null;
  document.getElementById("leg-genre").textContent =
    filtre ? (type === "c" ? C : G)[filtre[1]] : "Genre";
  dessiner(); });
if (!C.length) { afficherNoms = false; document.getElementById("choix-noms").style.display = "none"; }
document.getElementById("noms").addEventListener("change", e => { afficherNoms = e.target.checked; dessiner(); });
document.getElementById("reinit").addEventListener("click", () => {
  echelle = 1; decX = 0; decY = 0; choisir(-1); });

// Survol, clic, zoom, déplacement
let glisse = null, bouge = false;
canvas.addEventListener("pointerdown", e => { glisse = [e.offsetX, e.offsetY, decX, decY]; bouge = false;
  canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (glisse) { const dx = e.offsetX - glisse[0], dy = e.offsetY - glisse[1];
    if (Math.abs(dx) + Math.abs(dy) > 3) { bouge = true; canvas.classList.add("glisse"); }
    if (bouge) { decX = glisse[2] + dx; decY = glisse[3] + dy; bulle.style.display = "none"; dessiner(); return; } }
  const i = plusProche(e.offsetX, e.offsetY);
  if (i !== survol) { survol = i; dessiner(); }
  if (i >= 0) { bulle.innerHTML = details(A[i]); bulle.style.display = "block";
    const bx = Math.min(e.offsetX + 14, largeur - bulle.offsetWidth - 8);
    const by = e.offsetY + 14 + bulle.offsetHeight > hauteur ? e.offsetY - bulle.offsetHeight - 10 : e.offsetY + 14;
    bulle.style.left = bx + "px"; bulle.style.top = by + "px"; }
  else bulle.style.display = "none";
});
canvas.addEventListener("pointerup", e => { canvas.classList.remove("glisse");
  if (!bouge) { const i = plusProche(e.offsetX, e.offsetY); if (i >= 0) choisir(i, false); }
  glisse = null; });
canvas.addEventListener("pointerleave", () => { survol = -1; bulle.style.display = "none"; dessiner(); });
canvas.addEventListener("wheel", e => { e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015), nouvelle = Math.min(40, Math.max(0.5, echelle * f));
  decX = e.offsetX - (e.offsetX - decX) * nouvelle / echelle;
  decY = e.offsetY - (e.offsetY - decY) * nouvelle / echelle;
  echelle = nouvelle; dessiner(); }, { passive: false });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", dessiner);
new ResizeObserver(redimensionner).observe(cadre);
</script>
</body>
</html>
"""

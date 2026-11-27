"""Projection des profils de coefficients EASE, une position fixe par série."""

from hashlib import sha256

import altair as alt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def empreinte(poids, catalogue):
    signature = sha256(np.ascontiguousarray(poids).view(np.uint8))
    for valeurs in (catalogue.ids, catalogue.series):
        signature.update(np.asarray(valeurs, dtype="<i8").tobytes())
    return signature.hexdigest()


def projeter(poids, catalogue, coefficients=True):
    groupes, inverse = np.unique(catalogue.series, return_inverse=True)
    profils = np.zeros((len(groupes), poids.shape[1]), dtype=np.float32)
    np.add.at(profils, inverse, poids)
    profils /= np.bincount(inverse)[:, None]
    # Les liens entre saisons de la même série masqueraient les affinités entre séries.
    if coefficients:
        profils[inverse, np.arange(len(inverse))] = 0
    if len(groupes) < 3 or not profils.any():
        return groupes, np.column_stack((np.arange(len(groupes)), np.zeros(len(groupes))))
    with threadpool_limits(limits=2):
        reduits = TruncatedSVD(n_components=min(48, len(groupes) - 1),
                               random_state=42).fit_transform(normalize(profils))
        coords = TSNE(n_components=2, init="random", random_state=42,
                      perplexity=min(30, len(groupes) - 1), learning_rate="auto",
                      n_jobs=2).fit_transform(normalize(reduits))
    return groupes, coords


def positions(poids, catalogue, fichier):
    if fichier.is_file():
        with np.load(fichier, allow_pickle=False) as projection:
            if projection["signature"].item() == empreinte(poids, catalogue):
                return projection["groupes"], projection["coords"]
    return projeter(poids, catalogue)


def donnees_carte(catalogue, groupes, coords, etiquettes, vus, recommandations):
    vus = {catalogue.groupe(i) for i in vus}
    rangs = {catalogue.series[i]: (rang, etiquettes[raison])
             for rang, (i, raison) in enumerate(recommandations, 1)}
    lignes = []
    for groupe, (x, y) in zip(groupes, coords):
        rang, raison = rangs.get(groupe, (None, None))
        statut = "Recommended" if rang else "Watched" if groupe in vus else "Catalogue"
        lignes.append((x, y, etiquettes[catalogue.entrees[groupe]], statut, rang, raison))
    return pd.DataFrame(lignes, columns=["x", "y", "title", "status", "rank", "reason"])


def graphique(points, afficher_fond):
    domaine_x = [float(points.x.min()) - 1, float(points.x.max()) + 1]
    domaine_y = [float(points.y.min()) - 1, float(points.y.max()) + 1]
    if not afficher_fond:
        points = points[points.status != "Catalogue"]
    base = alt.Chart(points).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=domaine_x)),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=domaine_y)))
    statuts = ["Catalogue", "Watched", "Recommended"]
    couleur = alt.Color("status:N", title=None, scale=alt.Scale(
        domain=statuts, range=["#178578", "#1247dc", "#bd3000"]),
        legend=alt.Legend(orient="bottom", labelColor="#172033", symbolOpacity=1, symbolSize=110))
    forme = alt.Shape("status:N", title=None, scale=alt.Scale(
        domain=statuts, range=["circle", "diamond", "circle"]), legend=None)
    info = [alt.Tooltip("title:N", title="Series"), alt.Tooltip("status:N", title="Status")]
    info_rec = [*info, alt.Tooltip("rank:Q", title="Rank", format="d"),
                alt.Tooltip("reason:N", title="Strongest contribution")]
    couches = []
    for statut, taille, opacite in [("Catalogue", 21, 0.8), ("Watched", 145, 1), ("Recommended", 325, 1)]:
        details = info_rec if statut == "Recommended" else info
        couches.append(base.transform_filter(alt.datum.status == statut)
                       .mark_point(filled=True, size=taille, opacity=opacite, stroke="white",
                                   strokeWidth=0 if statut == "Catalogue" else 1.5)
                       .encode(color=couleur, shape=forme, tooltip=details))
    numeros = (base.transform_filter(alt.datum.status == "Recommended")
               .mark_text(color="white", fontSize=12, fontWeight="bold")
               .encode(text=alt.Text("rank:Q", format="d"), tooltip=info_rec))
    return (alt.layer(*couches, numeros).properties(height=450, background="#ffffff")
            .configure_view(stroke=None, fill="#ffffff").interactive())

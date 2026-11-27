"""Lancer avec python -m streamlit run src/demo.py."""

import streamlit as st

if __name__ == "__main__":
    st.set_page_config(page_title="Anime recommendations", page_icon="🎬", layout="wide")
    # Appliquer le style avant le chargement des bibliothèques et des modèles.
    st.html("""
    <style>
    [data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"] {
        display: block;
        width: 5rem;
        height: 0.4rem;
        margin: 0;
        overflow: hidden;
        border-radius: 1rem;
        background: #c3d3f5;
    }
    [data-testid="stStatusWidgetRunningIcon"] > * { display: none; }
    [data-testid="stStatusWidgetRunningIcon"]::after {
        content: "";
        display: block;
        height: 100%;
        width: 40%;
        border-radius: inherit;
        background: #1247dc;
        animation: anime-loading 1.2s ease-in-out infinite alternate;
    }
    @keyframes anime-loading {
        from { transform: translateX(0); }
        to { transform: translateX(150%); }
    }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stStatusWidgetRunningIcon"]::after { animation: none; }
    }
    </style>
    """)

import numpy as np
import pandas as pd

import config
import mal_profile as mal
import ease_map
from series import CatalogueSeries


@st.cache_resource
def callbacks():
    return mal.OAuthCallbacks()


@st.fragment(run_every=2)
def attendre_mal():
    state, verifier, url = st.session_state.mal_login
    try:
        reply = callbacks().take(state)
        if reply is None:
            st.link_button("Authorize on MyAnimeList ↗", url)
            st.caption("Keep this tab open. Your recommendations will appear here after authorization.")
            if st.button("Cancel sign-in"):
                callbacks().cancel(state)
                del st.session_state.mal_login
                st.rerun()
            return
        code, refused = reply
        if refused or not code:
            raise mal.ProfileError("Sign-in was cancelled. You can reconnect or import a list.")
        with st.spinner("Loading your anime list…"):
            st.session_state.mal_entries = mal.import_account(code, verifier)
    except mal.ProfileError as exc:
        st.session_state.mal_error = str(exc)
    del st.session_state.mal_login
    st.rerun()


def choisir_liste(catalogue):
    if "mal_error" in st.session_state:
        st.error(st.session_state.pop("mal_error"))
    if "mal_entries" not in st.session_state:
        if "mal_login" in st.session_state:
            attendre_mal()
            return [], [], {}
        if config.MAL_CLIENT_ID:
            if st.button("Connect my MyAnimeList account", type="primary"):
                st.session_state.mal_login = callbacks().start()
                st.rerun()
            with st.form("mal_public"):
                pseudo = st.text_input("Or import a public list by username")
                envoyer = st.form_submit_button("Load public list")
            if envoyer and pseudo.strip():
                try:
                    with st.spinner("Loading public anime list…"):
                        st.session_state.mal_entries = mal.download_list(username=pseudo)
                    st.rerun()
                except mal.ProfileError as exc:
                    st.error(str(exc))
        st.markdown("[Download your anime list from MyAnimeList](https://myanimelist.net/panel.php?go=export)")
        fichier = st.file_uploader("Import a MAL anime export (.xml or .xml.gz)", type=["xml", "gz"])
        if fichier is not None:
            try:
                st.session_state.mal_entries = mal.read_export(fichier.getvalue())
                st.rerun()
            except mal.ProfileError as exc:
                st.error(str(exc))
        return [], [], {}
    entries = st.session_state.mal_entries
    selection, vues, preferences, seuil = mal.rated_series_profile(entries, catalogue)
    st.caption(f"{len(entries):,} list entries · {len(preferences):,} series used for your profile · "
               f"{len(vues):,} previously watched series excluded")
    if seuil is not None:
        st.caption(f"Recommendations use series rated at or above your personal cutoff "
                   f"(about {seuil:.1f}/10). Ratings are averaged across seasons; higher ratings carry more weight.")
    if st.button("Forget this list"):
        del st.session_state.mal_entries
        st.rerun()
    if not selection:
        st.info("No eligible titles in this catalogue. Try choosing favourites manually.")
    return selection, vues, preferences


@st.cache_resource
def charger_modele(chemin, modification):
    with np.load(chemin, allow_pickle=False) as archive:
        poids = archive["weights"]
        ids = archive["mal_ids"]
        titres = archive["titles"]
        source = str(archive["source"].item())
    if poids.shape != (len(ids), len(ids)) or len(titres) != len(ids):
        raise ValueError("Model weights and catalogue have inconsistent dimensions.")
    return poids, ids, titres, source


@st.cache_resource
def charger_series(ids, chemin, modification):
    return CatalogueSeries(ids, chemin)


@st.cache_data
def charger_notes(fichier, modification):
    notes = pd.read_csv(fichier, usecols=["mal_id", "score"]).dropna(subset=["score"])
    return notes.set_index("mal_id")["score"].to_dict()


@st.cache_data
def charger_audiences(_catalogue, groupes, date_series, date_donnees):
    return _catalogue.audiences(config.ANIME_CSV)


@st.cache_resource
def charger_positions(_poids, _catalogue, modele, date_modele, date_series):
    fichier = config.RESULTS_DIR / "embeddings" / "espace_ease.npz"
    return ease_map.positions(_poids, _catalogue, fichier)


@st.cache_resource
def charger_comparaison(fichier, modification, ids, series, source):
    import scipy.sparse as sp
    with np.load(fichier, allow_pickle=False) as data:
        if (data["source"].item() != source or not np.array_equal(data["ids"], ids)
                or not np.array_equal(data["series"], series)):
            return None
        if "vecteurs" in data:
            vecteurs = data["vecteurs"]
            poids = vecteurs @ vecteurs.T
            np.fill_diagonal(poids, 0)
        else:
            poids = sp.csr_matrix((data["data"], data["indices"], data["indptr"]),
                                  shape=(len(ids), len(ids))).toarray()
        return poids, data["groupes"], data["coords"]


def afficher():
    if "code" in st.query_params or "error" in st.query_params:
        accepted = callbacks().receive(st.query_params.get("state", ""), st.query_params.get("code", ""),
                                       st.query_params.get("error", ""))
        st.query_params.clear()
        if accepted:
            st.info("Return to the original recommendations tab. You can close this tab.")
        else:
            st.error("This sign-in request expired or was already used. Reconnect in the original tab.")
        return
    st.title("What should I watch next?")
    chemin = config.MODELS_DIR / "ease_demo.npz"
    if not chemin.is_file():
        st.error("The local model is missing. Follow the model export instructions in the README.")
        st.stop()
    poids, ids, titres, source = charger_modele(chemin, chemin.stat().st_mtime_ns)
    fichier_series = config.CONTENT_DIR / "series.csv"
    catalogue = charger_series(ids, fichier_series, fichier_series.stat().st_mtime_ns)
    st.caption(f"MyAnimeList {source} · {len(catalogue.entrees):,} series and standalone titles")
    doublons = pd.Index(titres).duplicated(keep=False)
    etiquettes = [f"{titre} (MAL {ident})" if double else titre
                  for titre, ident, double in zip(titres, ids, doublons)]
    mode = st.radio("Recommendation source", ["Choose favourites", "MyAnimeList"], horizontal=True)
    if mode == "MyAnimeList":
        selection, exclus, preferences = choisir_liste(catalogue)
    else:
        selection = st.multiselect("Choose up to three series or standalone anime you enjoyed",
                                   sorted(catalogue.entrees.values(), key=lambda i: etiquettes[i].casefold()),
                                   format_func=etiquettes.__getitem__, max_selections=3,
                                   placeholder="Search for a title")
        exclus = ids[selection]
        preferences = None
        if not selection:
            st.info("Choose your first title to see ten recommendations.")
    audiences = charger_audiences(catalogue, tuple(catalogue.entrees), fichier_series.stat().st_mtime_ns,
                                  config.ANIME_CSV.stat().st_mtime_ns)
    originalite = st.slider("Originality · popular series to exclude", 0, 95,
                            value=25 if audiences else 0, step=5, format="%d%%", disabled=not audiences,
                            help="Move right to exclude more of the most popular series. At 0% the full catalogue is eligible.")
    exclusions_classement = set(exclus)
    if originalite and audiences:
        plafond = np.percentile(list(audiences.values()), 100 - originalite)
        exclusions_classement.update(g for g in catalogue.entrees
                                     if g not in audiences or audiences[g] > plafond)
        st.caption(f"Up to {int(plafond):,} MAL members on the most-followed entry of each series. "
                   "Eligible series are ranked by how well they match your profile.")
    notes = charger_notes(config.ANIME_CSV, config.ANIME_CSV.stat().st_mtime_ns)
    note_min = st.slider("Minimum community rating", 0.0, 10.0, 0.0, step=0.1, format="%.1f / 10",
                         help="Uses the community score of the entry shown in the results. "
                              "At 0 the filter is off. Above 0, unrated entries are excluded.")
    if note_min:
        exclusions_classement.update(g for g, i in catalogue.entrees.items()
                                     if notes.get(ids[i], 0) < note_min)
    st.subheader("Compare recommendation maps")
    fichiers = {"Item-KNN": config.MODELS_DIR / "comparison" / "item_knn.npz",
                "Synopsis (BGE)": config.MODELS_DIR / "comparison" / "bge.npz"}
    disponibles = ["EASE", *(nom for nom, fichier in fichiers.items() if fichier.is_file())]
    choix = st.multiselect("Methods to display", disponibles, default=disponibles[:2], max_selections=2)
    if not choix:
        st.info("Choose one or two methods to explore their maps.")
        return
    st.caption("Blue diamonds mark your history or favourites. Numbered orange circles mark each "
               "method’s own top picks. Hover for titles and scroll to zoom. Compare the neighbours "
               "around a series: each map has its own coordinate system.")
    fond = st.checkbox("Show the rest of the catalogue", value=True)
    descriptions = {"EASE": "Learned relationships between titles in viewing histories.",
                    "Item-KNN": "Neighbourhoods based on titles watched by the same people.",
                    "Synopsis (BGE)": "Neighbourhoods based on similar story descriptions."}
    classements = []
    for nom, colonne in zip(choix, st.columns(len(choix))):
        with colonne:
            st.subheader(nom)
            st.caption(descriptions[nom])
            if nom == "EASE":
                groupes, coords = charger_positions(poids, catalogue, chemin, chemin.stat().st_mtime_ns,
                                                    fichier_series.stat().st_mtime_ns)
                matrice = poids
            else:
                fichier = fichiers[nom]
                modele = charger_comparaison(fichier, fichier.stat().st_mtime_ns,
                                             ids, catalogue.series, source)
                if modele is None:
                    st.warning("This map uses another catalogue. Rebuild it with scripts/prepare_comparison_maps.py.")
                    continue
                matrice, groupes, coords = modele
            recommandations = catalogue.recommander(matrice, selection, exclusions_classement,
                                                     preferences=preferences) if selection else []
            points = ease_map.donnees_carte(catalogue, groupes, coords, etiquettes, exclus, recommandations)
            st.altair_chart(ease_map.graphique(points, fond), theme=None, key=f"map_{nom}")
            if recommandations:
                meilleurs, raisons = zip(*recommandations)
                classements.append(set(meilleurs))
                tableau = pd.DataFrame({"Rank": range(1, len(meilleurs) + 1),
                                        "Recommended anime": [etiquettes[i] for i in meilleurs],
                                        "MAL members": [audiences.get(catalogue.series[i]) for i in meilleurs],
                                        "Community rating / 10": [notes.get(ids[i]) for i in meilleurs],
                                        "Strongest contribution": [etiquettes[i] for i in raisons]})
                st.dataframe(tableau, hide_index=True)
            elif selection:
                st.info("No unseen series matched these settings. Try lowering the minimum rating or originality."
                        if originalite or note_min else "No unseen series received a positive score for this profile.")
    if len(classements) == 2:
        commun = len(classements[0] & classements[1])
        st.caption(f"Shared recommendations: {commun} of {len(classements[0])}.")


if __name__ == "__main__":
    afficher()

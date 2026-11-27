"""Régénérer data/content/series.csv depuis les relations publiques AniList."""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config

QUERY = """query($ids:[Int]) {
  Page(perPage:50) { media(type:ANIME, idMal_in:$ids) {
    idMal relations { edges { relationType node { type idMal } } }
  } }
}"""


def lire_lot(ids):
    for essai in range(8):
        r = requests.post(config.ANILIST_API_URL, json={"query": QUERY, "variables": {"ids": ids}},
                          timeout=45)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(int(r.headers.get("Retry-After") or min(60, 2 ** essai)) + 1)
            continue
        r.raise_for_status()
        reponse = r.json()
        if reponse.get("errors"):
            raise RuntimeError(reponse["errors"])
        return reponse["data"]["Page"]["media"]
    raise RuntimeError("AniList indisponible après plusieurs tentatives.")


def main():
    with config.ANIME_CSV.open(encoding="utf-8-sig", newline="") as source:
        catalogue = {int(row["mal_id"]): row for row in csv.DictReader(source)}
    ids = sorted(catalogue)
    cache = config.PROJECT_DIR / ".cache" / "series"
    cache.mkdir(parents=True, exist_ok=True)
    voisins = defaultdict(set)
    parents = defaultdict(set)
    def titres(i):
        row = catalogue.get(i, {})
        return {re.sub(r"[^\w]+", " ", row.get(champ, "").casefold()).strip()
                for champ in ("title_english", "title_romaji")} - {""}

    noms = {i: titres(i) for i in ids}

    def ajouter(a, b, relation):
        if relation in {"PREQUEL", "SEQUEL"}:
            voisins[a].add(b)
            voisins[b].add(a)
        elif relation in {"SIDE_STORY", "SUMMARY", "COMPILATION"}:
            parents[b].add(a)
        elif relation in {"PARENT", "PARENT_STORY", "FULL_STORY"}:
            parents[a].add(b)
        elif relation in {"ALTERNATIVE", "ALTERNATIVE_VERSION", "OTHER", "SPIN_OFF"} and a in noms and b in noms:
            # Une relation ambiguë ne suffit pas seule. Exiger le nom complet de la
            # série en tête du titre dérivé (ex. Jujutsu Kaisen Official PVs).
            for origine, derive in ((a, b), (b, a)):
                if any(len(x) >= 5 and y.startswith(x + " ") for x in noms[origine] for y in noms[derive]):
                    parents[derive].add(origine)
            if relation in {"ALTERNATIVE", "ALTERNATIVE_VERSION"}:
                formats = (catalogue[a]["format"], catalogue[b]["format"])
                if formats == ("TV", "Movie"):
                    parents[b].add(a)
                elif formats == ("Movie", "TV"):
                    parents[a].add(b)
    for debut in range(0, len(ids), 50):
        lot = ids[debut:debut + 50]
        fichier = cache / f"{lot[0]}-{lot[-1]}.json"
        sauvegarde = json.loads(fichier.read_text()) if fichier.exists() else {}
        if sauvegarde.get("ids") != lot:
            sauvegarde = {"ids": lot, "media": lire_lot(lot)}
            fichier.write_text(json.dumps(sauvegarde))
            time.sleep(2.1)
        for anime in sauvegarde["media"]:
            a = anime["idMal"]
            for lien in anime["relations"]["edges"]:
                node = lien["node"]
                b = node["idMal"]
                if not b or node["type"] != "ANIME":
                    continue
                ajouter(a, b, lien["relationType"])
    supplement = config.CONTENT_DIR / "series_relations_mal.csv"
    if supplement.exists():
        with supplement.open(encoding="utf-8", newline="") as fichier:
            for row in csv.DictReader(fichier):
                ajouter(int(row["mal_id"]), int(row["related_id"]), row["relation"].upper())

    # Certaines fiches MAL découpent une diffusion en parties, sans lien de suite.
    # Chercher le titre de base exact, en anglais ou en romaji, sans mot-clé flou.
    bases = defaultdict(set)
    for i in ids:
        if catalogue[i]["format"] in {"TV", "ONA", "OVA", "Movie"}:
            for nom in noms[i]:
                bases[nom].add(i)
    for i in ids:
        for nom in noms[i]:
            suffixe = re.search(r"\s+(?:(?:season|part|cour)\s+[1-9]\d*|[1-9]\d*(?:st|nd|rd|th)\s+season)\b", nom)
            if suffixe:
                candidats = bases.get(nom[:suffixe.start()], set()) - {i}
                if len(candidats) == 1:
                    parents[i].update(candidats)
    def ordre(i):
        row = catalogue[i]
        return (row["format"] not in {"TV", "ONA", "Movie"},
                float(row["start_year"] or 9999), float(row["start_month"] or 13),
                float(row["start_day"] or 32), i)
    chaine = {}
    for ident in list(voisins):
        if ident in chaine:
            continue
        membres = {ident}
        pile = [ident]
        while pile:
            for autre in voisins[pile.pop()] - membres:
                membres.add(autre)
                pile.append(autre)
        chaine.update(dict.fromkeys(membres, min(membres)))
    # Une histoire annexe liée à deux séries distinctes (crossover) ne les fusionne pas.
    rattachements = defaultdict(set)
    for enfant, origines in parents.items():
        c = chaine.get(enfant, enfant)
        for origine in origines:
            p = chaine.get(origine, origine)
            if p != c:
                rattachements[c].add(p)
    ensembles = defaultdict(set)
    for ident in set(chaine) | set(parents) | {p for ps in parents.values() for p in ps}:
        c = chaine.get(ident, ident)
        visites = {c}
        while len(rattachements[c]) == 1:
            suivant = next(iter(rattachements[c]))
            if suivant in visites:
                c = min(visites)
                break
            c = suivant
            visites.add(c)
        ensembles[c].add(ident)
    groupes = {}
    for c, membres in ensembles.items():
        if len(membres) < 2 or not membres.intersection(catalogue):
            continue
        presents = sorted(membres.intersection(catalogue),
                          key=lambda i: (chaine.get(i, i) != c, ordre(i)))
        # Le rang permet de choisir un point d'entrée même si la première saison
        # du catalogue de contenu est absente du modèle collaboratif.
        rangs = {i: n for n, i in enumerate(presents)}
        for i in membres:
            groupes[i] = (presents[0], rangs.get(i, len(presents)))
    destination = config.CONTENT_DIR / "series.csv"
    temporaire = destination.with_suffix(".tmp")
    with temporaire.open("w", encoding="utf-8", newline="") as sortie:
        writer = csv.writer(sortie)
        writer.writerow(["mal_id", "series_id", "entry_order"])
        writer.writerows((i, *groupes[i]) for i in sorted(groupes))
    temporaire.replace(destination)
    print(f"{len(groupes)} titres reliés dans {len(set(g for g, _ in groupes.values()))} séries.")


if __name__ == "__main__":
    main()

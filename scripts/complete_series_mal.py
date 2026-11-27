"""Compléter les relations AniList absentes avec les relations anime de MAL."""

import csv
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config


def main():
    cache = config.PROJECT_DIR / ".cache" / "series"
    lots = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    with config.ANIME_CSV.open(encoding="utf-8-sig") as fichier:
        ids = {int(r["mal_id"]) for r in csv.DictReader(fichier)}
    if ids != {i for lot in lots for i in lot["ids"]}:
        raise RuntimeError("Exécuter prepare_series.py pour obtenir le catalogue AniList complet.")
    manquants = ids - {a["idMal"] for lot in lots for a in lot["media"]}
    sortie = config.CONTENT_DIR / "series_relations_mal.csv"
    lignes = []
    with requests.Session() as session:
        session.headers["X-MAL-CLIENT-ID"] = config.MAL_CLIENT_ID
        for i in sorted(manquants):
            reponse = session.get(f"{config.MAL_API_URL}/anime/{i}", params={"fields": "related_anime"},
                                  timeout=30)
            reponse.raise_for_status()
            for lien in reponse.json()["related_anime"]:
                lignes.append((i, lien["relation_type"], lien["node"]["id"]))
            time.sleep(0.3)
    tmp = sortie.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(["mal_id", "relation", "related_id"])
        writer.writerows(sorted(lignes))
    tmp.replace(sortie)
    print(len(manquants), "titres consultés,", len(lignes), "relations.")


if __name__ == "__main__":
    main()

"""Inspecter le dataset MAL 2025 : structure et statistiques de ratings.csv et favs.csv.

Dataset Kaggle neelagiriaditya/anime-dataset-jan-1917-to-oct-2025 : sur Kaggle il est
monté en entrée (DMANIME_MAL2025_DIR) ; en local, data/raw/mal-2025/datasets/.
"""

import duckdb

import config

DESCRIPTION = "Inspecter ratings.csv et favs.csv du dataset MAL 2025 (colonnes, statistiques)"
FICHIERS = ("ratings.csv", "favs.csv")


def ajouter_arguments(parser):
    pass


def executer(args):
    dossier = config.MAL2025_DIR
    print(f"Dossier : {dossier}")
    for nom_fichier in FICHIERS:
        chemin = dossier / nom_fichier
        print(f"\n=== {nom_fichier} ({chemin.stat().st_size / 1e6:.1f} Mo)")
        source = (f"read_csv('{chemin.as_posix()}', auto_detect=true, "
                  "sample_size=200000, ignore_errors=true)")
        print(duckdb.sql(f"SELECT * FROM {source} LIMIT 5").df().to_string()[:2000])
        colonnes = duckdb.sql(f"DESCRIBE SELECT * FROM {source}").df()
        print("lignes :", duckdb.sql(f"SELECT count(*) FROM {source}").fetchone()[0])
        for _, ligne in colonnes.iterrows():
            nom, type_ = ligne.column_name, ligne.column_type
            n, mini, maxi = duckdb.sql(f'SELECT count(DISTINCT "{nom}"), min("{nom}"), '
                                       f'max("{nom}") FROM {source}').fetchone()
            print(f"  {nom:<24} {type_:<10} distincts={n:,} min={mini} max={maxi}")
            if n <= 20:
                print("    ", duckdb.sql(f'SELECT "{nom}", count(*) FROM {source} '
                                       "GROUP BY 1 ORDER BY 2 DESC").fetchall())

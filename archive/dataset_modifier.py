import csv
from pathlib import Path

source = Path("archive/anime_dataset.csv")
destination = Path("archive/anime_synopsis.csv")

with source.open("r", encoding="utf-8-sig", newline="") as entree, \
     destination.open("w", encoding="utf-8", newline="") as sortie:
    lecteur = csv.DictReader(entree)
    ecrivain = csv.DictWriter(sortie, fieldnames=["mal_id", "synopsis"])
    ecrivain.writeheader()

    for ligne in lecteur:
        ecrivain.writerow({
            "mal_id": ligne["mal_id"],
            "synopsis": ligne["description"],
        })

print(f"Nouveau dataset créé : {destination}")

import csv
import re
from pathlib import Path

source = Path("archive/anime_dataset.csv")
destination = Path("archive/anime_synopsis.train")

with source.open("r", encoding="utf-8-sig", newline="") as entree, \
     destination.open("w", encoding="utf-8") as sortie:
    lecteur = csv.DictReader(entree)

    for ligne in lecteur:
        synopsis = ligne["description"]

        # Supprimer la mention finale, avec son éventuel guillemet.
        synopsis = re.sub(
            r'\s*\[Written by MAL Rewrite\]\s*"?\s*$',
            "",
            synopsis,
        )

        # Supprimer les virgules et les guillemets doubles.
        synopsis = synopsis.replace(",", "").replace('"', "")

        # Garder chaque synopsis sur une seule ligne.
        synopsis = " ".join(synopsis.split())

        sortie.write(f'{ligne["mal_id"]} {synopsis}\n')

print(f"Fichier créé : {destination}")
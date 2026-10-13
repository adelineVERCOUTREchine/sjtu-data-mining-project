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
from pathlib import Path

source = Path("archive/anime_dataset.csv")
destination = Path("archive/anime_synopsis.train")

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

print(f"Fichier créé : {destination}")
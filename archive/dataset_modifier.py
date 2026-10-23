import csv
import re
import string
import unicodedata
from pathlib import Path


def nettoyer_synopsis(texte):
    # Retirer les crédits avant de remplacer leurs délimiteurs.
    texte = re.sub(
        r"\[\s*(?:Writ{2,3}en\s+by\b|Source\s*:)[^\]]*\]"
        r"|\(\s*(?:Writ{2,3}en\s+by\b|Source\s*:)"
        r"(?:[^()]|\([^()]*\))*\)",
        " ",
        texte,
        flags=re.IGNORECASE,
    )
    texte = texte.lower()
    texte = "".join(
        " " if caractere in string.punctuation
        or unicodedata.category(caractere).startswith("P") else caractere
        for caractere in texte
    )
    return " ".join(texte.split())


def main():
    archive = Path(__file__).resolve().parent
    source = archive / "anime_dataset.csv"
    destination_csv = archive / "anime_synopsis.csv"
    destination_train = archive / "anime_synopsis.train"
    conservees = 0
    supprimees = 0

    with source.open("r", encoding="utf-8-sig", newline="") as entree, \
         destination_csv.open("w", encoding="utf-8", newline="") as sortie_csv, \
         destination_train.open("w", encoding="utf-8", newline="") as sortie_train:
        lecteur = csv.DictReader(entree)
        ecrivain = csv.DictWriter(sortie_csv, fieldnames=["mal_id", "synopsis"])
        ecrivain.writeheader()

        for ligne in lecteur:
            description = ligne["description"]
            synopsis = nettoyer_synopsis(description)

            # Conserver le filtre initial et vérifier aussi le texte nettoyé.
            if len(description.split()) < 3 or len(synopsis.split()) < 3:
                supprimees += 1
                continue

            ecrivain.writerow({"mal_id": ligne["mal_id"], "synopsis": synopsis})
            sortie_train.write(f'{ligne["mal_id"]} {synopsis}\n')
            conservees += 1

    print(f"Fichiers créés : {destination_csv} et {destination_train}")
    print(f"Bilan : {conservees + supprimees} entrées lues, "
          f"{conservees} conservées, {supprimees} supprimées "
          "(synopsis de moins de 3 mots avant ou après nettoyage).")


if __name__ == "__main__":
    main()

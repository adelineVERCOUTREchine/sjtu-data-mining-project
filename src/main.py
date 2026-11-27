"""Lanceur du projet : exécute une ou plusieurs étapes / méthodes, dans l'ordre donné.

Exemples :
    python src/main.py --list
    python src/main.py prepare_interactions
    python src/main.py popularity item_knn ease
    python src/main.py ease --lam 300 --split test
    python src/main.py minhash_genres_tags --threshold 0.3 --num-hashes 256
    python src/main.py ease --help

Les options placées après les noms sont transmises à chaque méthode sélectionnée ;
une option inconnue de toutes les méthodes provoque une erreur.

Ajouter une méthode : créer src/methods/<nom>.py avec DESCRIPTION,
ajouter_arguments(parser) et executer(args), puis l'ajouter à METHODES ci-dessous.
"""

import argparse
import importlib
import sys
import time

PREPARATION = {
    "download_interactions": "preparation.download_interactions",
    "prepare_content": "preparation.prepare_content",
    "prepare_interactions": "preparation.prepare_interactions",
    "inspect_mal2025": "preparation.inspect_mal2025",
}

METHODES = {
    # Similarité de contenu entre animes (item-to-item)
    "minhash_synopsis": "methods.minhash_synopsis",
    "minhash_genres_tags": "methods.minhash_genres_tags",
    "minhash_genres_tags_synopsis": "methods.minhash_genres_tags_synopsis",
    "synopsis_embeddings": "methods.synopsis_embeddings",
    "embeddings_map": "methods.embeddings_map",
    "synopsis_finetune": "methods.synopsis_finetune",
    # Recommandation à partir des listes MAL (évaluées sur le même protocole)
    "popularity": "methods.popularity",
    "item_knn": "methods.item_knn",
    "ease": "methods.ease",
    "funnel": "methods.funnel",
}

TOUT = {**PREPARATION, **METHODES}


def afficher_liste():
    for titre, groupe in (("Préparation des données", PREPARATION), ("Méthodes", METHODES)):
        print(f"\n{titre} :")
        for nom, chemin in groupe.items():
            print(f"  {nom:<30} {importlib.import_module(chemin).DESCRIPTION}")
    print("\nAide d'une méthode : python src/main.py <nom> --help")


def creer_parser(nom, module):
    parser = argparse.ArgumentParser(prog=f"main.py {nom}", description=module.__doc__,
                                     allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    module.ajouter_arguments(parser)
    return parser


def main(argv=None):
    # Accents et symboles lisibles même quand la sortie est redirigée (Windows).
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    noms = []
    while argv and not argv[0].startswith("-"):
        noms.append(argv.pop(0))
    options = argv

    if not noms:
        if options and options[0] not in ("--list", "-l", "--help", "-h"):
            sys.exit(f"Option inattendue sans méthode : {options[0]}")
        print(__doc__)
        afficher_liste()
        return

    inconnus = [nom for nom in noms if nom not in TOUT]
    if inconnus:
        sys.exit(f"Inconnu : {', '.join(inconnus)}. Liste : python src/main.py --list")

    # Analyser toutes les options avant d'exécuter quoi que ce soit.
    taches = []
    non_reconnues = None
    for nom in noms:
        module = importlib.import_module(TOUT[nom])
        args, reste = creer_parser(nom, module).parse_known_args(options)
        reste = {option for option in reste if option.startswith("-")}
        non_reconnues = reste if non_reconnues is None else non_reconnues & reste
        taches.append((nom, module, args))
    if non_reconnues:
        sys.exit(f"Option(s) reconnue(s) par aucune méthode : {', '.join(sorted(non_reconnues))}")

    for nom, module, args in taches:
        print(f"\n{'=' * 70}\n▶ {nom}\n{'=' * 70}", flush=True)
        debut = time.perf_counter()
        module.executer(args)
        print(f"✔ {nom} terminé en {time.perf_counter() - debut:.1f} s.", flush=True)


if __name__ == "__main__":
    main()

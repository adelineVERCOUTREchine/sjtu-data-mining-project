"""Exécuter des méthodes sur Kaggle (GPU) au lieu du PC, puis rapatrier les résultats.

Usage (mêmes arguments que src/main.py) :
    python scripts/kaggle_run.py synopsis_embeddings --model BAAI/bge-small-en-v1.5
    python scripts/kaggle_run.py popularity item_knn ease --split test
    python scripts/kaggle_run.py --cpu ease          # sans GPU (quota GPU préservé)
    python scripts/kaggle_run.py --no-wait ease      # lancer sans attendre la fin
    python scripts/kaggle_run.py --fetch             # récupérer la dernière exécution

Étapes :
  1. Le code (src/, data/content/, requirements.txt) est envoyé comme dataset
     Kaggle privé <utilisateur>/dmanime-code (nouvelle version à chaque lancement).
  2. Un notebook privé <utilisateur>/dmanime-run l'exécute avec le dataset des
     listes MAL monté en entrée ; prepare_interactions est ajouté automatiquement
     si une méthode évaluée en a besoin.
  3. Ses sorties (results/, models/) sont téléchargées dans le projet ; les
     lignes de results/recommandation/resultats.csv sont ajoutées au tableau local.

Nécessite KAGGLE_USERNAME et KAGGLE_KEY dans .env.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_DIR / ".kaggle_build"
sys.path.insert(0, str(PROJECT_DIR / "src"))

import config  # noqa: E402  (charge aussi .env)

CODE_SLUG = "dmanime-code"
KERNEL_SLUG = "dmanime-run"
GPU = "NvidiaTeslaT4"
# Méthodes sans évaluation sur les listes MAL (pas besoin de prepare_interactions).
SANS_INTERACTIONS = {"download_interactions", "prepare_content", "prepare_interactions",
                     "minhash_synopsis", "minhash_genres_tags", "minhash_genres_tags_synopsis",
                     "embeddings_map", "inspect_mal2025"}

RUNNER = r'''
"""Généré par scripts/kaggle_run.py : exécute src/main.py sur Kaggle."""
import importlib.util, os, shutil, subprocess, sys
from pathlib import Path

ARGS = __ARGS__
PAQUETS = {"duckdb": "duckdb", "implicit": "implicit", "dotenv": "python-dotenv",
           "sentence_transformers": "sentence-transformers", "lightgbm": "lightgbm",
           "umap": "umap-learn", "datasets": "datasets", "accelerate": "accelerate"}

code = next(Path("/kaggle/input").rglob("src/main.py")).parent.parent
projet = Path("/kaggle/tmp/projet")
shutil.copytree(code, projet, dirs_exist_ok=True)
brut = next(Path("/kaggle/input").rglob("animelist.csv")).parent
os.environ["DMANIME_RAW_DIR"] = str(brut)
mal2025 = [p.parent for p in Path("/kaggle/input").rglob("ratings.csv") if (p.parent / "details.csv").exists()]
if mal2025:
    os.environ["DMANIME_MAL2025_DIR"] = str(mal2025[0])
    print("MAL 2025 :", mal2025[0], flush=True)
# Un seul GPU : avec deux T4, PyTorch duplique le modèle (DataParallel) et sature la mémoire.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print("Code :", code, "| Listes MAL :", brut, flush=True)

manquants = [p for m, p in PAQUETS.items() if importlib.util.find_spec(m) is None]
if manquants:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *manquants], check=True)

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
retour = subprocess.run([sys.executable, "src/main.py", *ARGS], cwd=projet).returncode

for dossier in ("results", "models"):
    if (projet / dossier).exists():
        shutil.copytree(projet / dossier, Path("/kaggle/working") / dossier, dirs_exist_ok=True)
sys.exit(retour)
'''


def api_kaggle():
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def preparer_code(utilisateur):
    dossier = BUILD_DIR / "code"
    shutil.rmtree(dossier, ignore_errors=True)
    ignorer = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(PROJECT_DIR / "src", dossier / "src", ignore=ignorer)
    shutil.copytree(config.CONTENT_DIR, dossier / "data" / "content", ignore=ignorer)
    shutil.copy(PROJECT_DIR / "requirements.txt", dossier)
    # Embeddings déjà calculés : évite de ré-encoder les synopsis sur Kaggle.
    embeddings = config.MODELS_DIR / "embeddings"
    if embeddings.exists():
        shutil.copytree(embeddings, dossier / "models" / "embeddings", ignore=ignorer)
    (dossier / "dataset-metadata.json").write_text(json.dumps({
        "title": CODE_SLUG, "id": f"{utilisateur}/{CODE_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }), encoding="utf-8")
    return dossier


def envoyer_code(api, utilisateur):
    dossier = preparer_code(utilisateur)
    ref = f"{utilisateur}/{CODE_SLUG}"
    try:
        api.dataset_status(ref)
        existe = True
    except Exception:
        existe = False
    print(f"{'Mise à jour' if existe else 'Création'} du dataset privé {ref}...", flush=True)
    if existe:
        api.dataset_create_version(str(dossier), version_notes=time.strftime("%Y-%m-%d %H:%M"),
                                   quiet=True, dir_mode="zip")
    else:
        api.dataset_create_new(str(dossier), public=False, quiet=True, dir_mode="zip")
    for _ in range(60):
        time.sleep(5)
        try:
            if str(api.dataset_status(ref)).lower().endswith("ready"):
                return ref
        except Exception:
            pass  # Juste après l'envoi, Kaggle répond parfois 403 le temps du traitement.
    raise TimeoutError(f"Le dataset {ref} n'est pas prêt après 5 minutes.")


def envoyer_notebook(api, utilisateur, code_ref, arguments, gpu):
    dossier = BUILD_DIR / "kernel"
    shutil.rmtree(dossier, ignore_errors=True)
    dossier.mkdir(parents=True)
    (dossier / "run.py").write_text(RUNNER.replace("__ARGS__", repr(arguments)), encoding="utf-8")
    metadata = {
        "id": f"{utilisateur}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [code_ref, config.KAGGLE_INTERACTIONS, config.KAGGLE_MAL2025],
    }
    if gpu:
        metadata["machine_shape"] = GPU
    (dossier / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reponse = api.kernels_push(str(dossier))
    erreur = getattr(reponse, "error", None)
    if erreur:
        raise RuntimeError(f"Envoi du notebook refusé : {erreur}")
    print(f"Notebook lancé : https://www.kaggle.com/code/{utilisateur}/{KERNEL_SLUG}", flush=True)
    return metadata["id"]


def attendre(api, kernel):
    debut = time.time()
    etat = ""
    while True:
        time.sleep(20)
        statut = api.kernels_status(kernel)
        etat = str(getattr(statut, "status", statut)).lower()
        print(f"  [{time.time() - debut:>5.0f} s] {etat}", flush=True)
        if any(fin in etat for fin in ("complete", "error", "cancel")):
            return etat, getattr(statut, "failure_message", "") or ""


def rapatrier(api, kernel):
    sortie = BUILD_DIR / "output"
    shutil.rmtree(sortie, ignore_errors=True)
    sortie.mkdir(parents=True)
    api.kernels_output(kernel, str(sortie), force=True, quiet=True)

    journal = next(sortie.glob("*.log"), None)
    tableau_distant = sortie / "results" / "recommandation" / "resultats.csv"
    if tableau_distant.exists():
        with tableau_distant.open(encoding="utf-8", newline="") as entree:
            lignes = list(csv.DictReader(entree))
        tableau = config.RESULTS_DIR / "recommandation" / "resultats.csv"
        tableau.parent.mkdir(parents=True, exist_ok=True)
        nouveau = not tableau.exists()
        with tableau.open("a", encoding="utf-8", newline="") as fichier:
            for ligne in lignes:
                ligne["methode"] = ligne["methode"] + " (kaggle)"
                ecrivain = csv.DictWriter(fichier, fieldnames=list(ligne))
                if nouveau:
                    ecrivain.writeheader()
                    nouveau = False
                ecrivain.writerow(ligne)
        tableau_distant.unlink()
        print(f"{len(lignes)} ligne(s) ajoutée(s) à {tableau}")

    for dossier, destination in (("results", config.RESULTS_DIR), ("models", config.MODELS_DIR)):
        if (sortie / dossier).exists():
            shutil.copytree(sortie / dossier, destination, dirs_exist_ok=True)
            print(f"Sorties copiées dans {destination}")
    return journal


def main():
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="ne pas demander de GPU")
    parser.add_argument("--no-wait", action="store_true", help="ne pas attendre la fin")
    parser.add_argument("--fetch", action="store_true",
                        help="attendre puis récupérer la dernière exécution, sans rien relancer")
    options, arguments = parser.parse_known_args()
    if options.fetch:
        for flux in (sys.stdout, sys.stderr):
            flux.reconfigure(encoding="utf-8")
        api = api_kaggle()
        kernel = f"{api.get_config_value('username')}/{KERNEL_SLUG}"
        print(f"Attente de {kernel}...", flush=True)
        etat, message = attendre(api, kernel)
        journal = rapatrier(api, kernel)
        print(f"Journal complet : {journal}" if journal else "Pas de journal.")
        if "complete" not in etat:
            sys.exit(f"Exécution Kaggle terminée en échec ({etat}) {message}")
        print("Résultats récupérés.")
        return
    if not arguments:
        parser.error("indiquer au moins une méthode (voir python src/main.py --list)")
    for flux in (sys.stdout, sys.stderr):
        flux.reconfigure(encoding="utf-8")

    # Comme main.py : les noms de méthodes précèdent la première option.
    noms = []
    for argument in arguments:
        if argument.startswith("-"):
            break
        noms.append(argument)
    collaboratif = "collaboratif" in arguments and "embeddings_map" in noms
    if "prepare_interactions" not in noms and (set(noms) - SANS_INTERACTIONS or collaboratif):
        arguments = ["prepare_interactions", *arguments]

    api = api_kaggle()
    utilisateur = api.get_config_value("username")
    code_ref = envoyer_code(api, utilisateur)
    kernel = envoyer_notebook(api, utilisateur, code_ref, arguments, not options.cpu)
    if options.no_wait:
        print(f"Récupérer plus tard : kaggle kernels output {kernel} -p .kaggle_build/output")
        return

    etat, message = attendre(api, kernel)
    journal = rapatrier(api, kernel)
    if journal:
        print(f"Journal complet : {journal}")
    if "complete" not in etat:
        sys.exit(f"Exécution Kaggle terminée en échec ({etat}) {message}")
    print("Exécution Kaggle terminée.")


if __name__ == "__main__":
    main()

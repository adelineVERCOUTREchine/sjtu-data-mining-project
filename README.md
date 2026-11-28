# Recommandation d'animés à partir d'un profil MyAnimeList ou d'un animé choisi

Projet de data mining (SJTU AI3602, automne 2026). On compare des méthodes de
**filtrage collaboratif** (ce que regardent les utilisateurs) et de **contenu**
(synopsis, genres, tags) pour deux tâches :

- **à partir d'un animé** : « j'ai aimé *Death Note*, que regarder ensuite ? » ;
- **à partir d'un profil MAL** : recommander des animés à un utilisateur d'après sa liste.

Documentation : [plan et état de l'art](docs/methodes_recommandation.md) ·
[consignes du cours](docs/consignes_projet.md) · [méthodes MinHash](docs/minhash.md) ·
[idées initiales](docs/idees_initiales.md)

## Installation

Python 3.12, sans venv :

```powershell
python -m pip install -r requirements.txt
copy .env.example .env          # renseigner MAL_CLIENT_ID et KAGGLE_API_TOKEN
python scripts/check_env.py     # vérifie modules, GPU, catalogue et clés
```

## Utilisation

### Démo interactive locale

Depuis la racine du projet, lancer la démo en anglais avec le modèle déjà exporté :

```powershell
python -m streamlit run src/demo.py --server.address 127.0.0.1
```

Dans l’onglet **More Like My Favourites**, choisir un à trois titres. Le classement des dix suggestions se recalcule à chaque
modification. Pour chaque suggestion, la dernière colonne indique le titre choisi
dont le coefficient EASE contribue le plus au score. Les titres sélectionnés sont exclus.
Le modèle local `models/ease_demo.npz` contient les poids,
les titres dans l'ordre des colonnes, les identifiants MAL et l'édition des données.
Ce fichier est local et ignoré par Git. Pour déplacer la démo, le copier au même
emplacement avec le code. Le calcul des suggestions fonctionne hors ligne.

Le sélecteur et le top 10 présentent une entrée par série. La table locale
`data/content/series.csv` associe les identifiants MAL à partir des relations
AniList entre suites et préquelles, complétées par les relations MyAnimeList
pour les fiches absentes et par les titres de saison dont la base est identifiable
sans ambiguïté. Elle rattache aussi les histoires annexes et
compilations dont la série parente est identifiable. Un titre sans lien connu
garde sa propre entrée. Le script `scripts/prepare_series.py` construit cette table.

Chaque série déjà entamée ou abandonnée est exclue intégralement. Pour le classement,
les saisons positives d'une même série contribuent par leur moyenne. Le meilleur
score d'une saison détermine le rang de la série candidate, et le résultat affiche
son point d'entrée disponible dans le modèle.

Dans **Compare recommendation maps**, sélectionner une ou deux méthodes. Chaque carte
montre le même profil, son propre classement et les séries communes aux deux listes
sont comptées sous les tableaux. Les coordonnées des cartes sont indépendantes.
Le curseur **Originality** écarte les séries les plus populaires avant de choisir
le top de chaque méthode. À 0 %, tout le catalogue est éligible. À 75 %, seuls
les titres du quart le moins populaire (et les ex æquo au seuil) sont éligibles.
Le réglage initial est de 25 %. L'audience d'une série est le nombre de membres
MAL de sa fiche la plus suivie dans le catalogue local.
**Item-KNN** repose sur les co-visionnages (100 voisins, régularisation 10) et
**Synopsis (BGE)** sur la similarité cosinus des embeddings de synopsis.
Pour ces deux méthodes, copier aussi `models/comparison/` sur le portable ou
exécuter `python scripts/prepare_comparison_maps.py` avec les données locales
et les embeddings du projet. Les deux cartes fonctionnent ensuite hors ligne.

La carte **EASE** place chaque série à partir de la moyenne des lignes
de la matrice EASE de ses saisons. Les coefficients internes à la série sont mis
à zéro, puis les vecteurs normalisés sont réduits par SVD (48 dimensions) et
projetés par t-SNE en 2D. La proximité donne une indication locale de similarité
entre profils de recommandation. Le classement est calculé dans le modèle complet.
Les positions restent fixes quand le profil change. Les losanges bleus identifient
les séries vues ou choisies et les numéros orange correspondent au top affiché.
La projection locale `results/embeddings/espace_ease.npz` est utilisée si sa
signature correspond au modèle et au regroupement des saisons. Sinon la démo
recalcule les positions et les garde en mémoire.

Le bouton **Anime Advisor** ouvre un panneau conversationnel en anglais.
Il utilise DeepSeek via ArcBench (`https://api.arc-bench.com/v1`, modèle
`deepseek-flash`, `reasoning_effort=low`). Saisir la clé ArcBench dans le champ
masqué du panneau, ou définir `ARCBENCH_API_KEY` dans `.env` puis redémarrer.
Les cartes et les listes fonctionnent également sans le service de chat.

À chaque question, le conseiller interprète la demande, effectue une recherche
locale dans tout le catalogue admissible et combine les résultats avec ceux des
trois méthodes disponibles, indépendamment des cartes affichées. Une deuxième
requête produit la réponse à partir des fiches retenues, des rangs, des mesures
de validation et d'un résumé des préférences. Les requêtes transmettent à ArcBench
la question et le contexte récent de la conversation. Les identifiants de compte
et jetons MAL restent locaux. Changer de profil efface l'historique du panneau.

Dans l’onglet **MyAnimeList Recommendations**, deux options permettent de charger le profil :

- **Export du compte MAL** — Cliquer sur **Connect my MyAnimeList account** pour ouvrir la
  page d'export MAL, choisir **Anime List**, puis importer le fichier `.xml.gz` ou `.xml` dans la démo.
- **Liste publique** — Entrer le pseudo MAL et cliquer sur **Load public list**.
  Cette option utilise `MAL_CLIENT_ID` dans `.env`.

Pour la démo, les notes des saisons terminées, en cours ou en pause donnent une
moyenne par série. Le seuil personnel est la moyenne des séries notées représentées
dans le catalogue, avec un minimum de 6/10. Chaque série qui atteint le seuil
contribue selon le poids `(note moyenne − 5) / 5`, y compris pour l'explication
« Strongest contribution ». Sans note exploitable, les visionnages alimentent
le profil à poids égal. Tous les titres de ces trois statuts et les
titres abandonnés sont exclus des suggestions. Les titres prévus restent éligibles.
La liste importée reste en mémoire dans la session jusqu'à **Forget this list** ou
la fin de la session. Le jeton OAuth sert uniquement au chargement initial.

Pour reproduire l'export et les deux évaluations à partir du fichier
`data/raw/mal-2025/datasets/ratings.csv` de l'édition 2025 :

```powershell
$env:DMANIME_PROCESSED_DIR = Join-Path $PWD 'data/processed/2025'
python src/main.py prepare_interactions --source 2025 --memory-limit 2GB
python src/main.py popularity ease --split test --lam 500 --save-demo --batch-size 256 --exemple 0
```

### Méthodes en ligne de commande

`src/main.py` exécute les étapes de préparation et les méthodes demandées dans l'ordre :

```powershell
python src/main.py --list                                     # étapes et méthodes
python src/main.py download_interactions prepare_interactions # une fois : listes MAL -> matrices
python src/main.py popularity item_knn ease                   # évaluer plusieurs méthodes
python src/main.py ease --lam 300 --split test                # options transmises aux méthodes
python src/main.py ease --help                                # options d'une méthode
python -B -m unittest discover -s tests -v                    # tests
```

**Calculs lourds sur Kaggle** (GPU T4 gratuit) : mêmes arguments ; le script envoie
le code, exécute un notebook privé et rapatrie `results/` et `models/` :

```powershell
python scripts/kaggle_run.py synopsis_finetune                 # GPU (file d'attente possible)
python scripts/kaggle_run.py --cpu embeddings_map              # sans GPU : pas d'attente
python scripts/kaggle_run.py --fetch                           # récupérer la dernière exécution
```

`prepare_interactions` est ajouté automatiquement quand une méthode a besoin des
listes MAL. Les embeddings déjà calculés (`models/embeddings/`) sont envoyés avec
le code pour ne pas être ré-encodés.

## Méthodes

| Nom | Type | Tâche | Fichier |
| --- | --- | --- | --- |
| `minhash_synopsis` | Contenu : shingles de 3 mots | Paires d'animés similaires | [minhash_synopsis.py](src/methods/minhash_synopsis.py) |
| `minhash_genres_tags` | Contenu : genres et tags | Paires d'animés similaires | [minhash_genres_tags.py](src/methods/minhash_genres_tags.py) |
| `minhash_genres_tags_synopsis` | Contenu : les deux | Paires d'animés similaires | [minhash_genres_tags_synopsis.py](src/methods/minhash_genres_tags_synopsis.py) |
| `synopsis_embeddings` | Contenu : Transformer (bge-base) + cosinus | Animé et profil | [synopsis_embeddings.py](src/methods/synopsis_embeddings.py) |
| `synopsis_finetune` | Contenu : Transformer affiné sur les co-visionnages | Animés jamais regardés (cold-start) | [synopsis_finetune.py](src/methods/synopsis_finetune.py) |
| `embeddings_map` | Visualisation UMAP | Carte interactive HTML | [embeddings_map.py](src/methods/embeddings_map.py) |
| `popularity` | Baseline | Profil | [popularity.py](src/methods/popularity.py) |
| `item_knn` | Filtrage collaboratif : cosinus item-item | Animé et profil | [item_knn.py](src/methods/item_knn.py) |
| `ease` | Filtrage collaboratif : EASE^R (Steck, 2019) | Animé et profil | [ease.py](src/methods/ease.py) |
| `funnel` | Entonnoir : retrieval (EASE, KNN, contenu, popularité) + re-ranking LightGBM LambdaMART | Profil | [funnel.py](src/methods/funnel.py) |

**Ajouter une méthode** : créer `src/methods/<nom>.py` avec `DESCRIPTION`,
`ajouter_arguments(parser)` et `executer(args)`, puis l'enregistrer dans `METHODES`
([src/main.py](src/main.py)). Une méthode de recommandation entraîne son modèle puis
appelle `lancer_evaluation(nom, scorer, args, params)`, où `scorer(X)` renvoie les
scores des animés pour les utilisateurs de la matrice creuse `X`.

## Données

| Source | Contenu |
| --- | --- |
| [data/content/anime_dataset.csv](data/content/anime_dataset.csv) | Catalogue de 4 985 animés populaires (AniList, identifiants MAL) : titres, synopsis, genres, tags, studios, scores |
| **Kaggle [*Anime Dataset 2025*](https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025)** (défaut, CC BY-NC-SA 4.0) | Snapshot MAL d'**octobre 2025** : 124 M lignes de listes (309 k utilisateurs), 28 955 fiches, 105 k recommandations d'utilisateurs MAL |
| Kaggle *Anime Recommendation Database 2020* (`--source 2020`) | 109 M lignes de listes MAL, 325 770 utilisateurs ; les animés sortis après 2020 en sont absents |

Sur Kaggle, les deux datasets sont montés automatiquement (`scripts/kaggle_run.py`) ;
en local, `download_interactions` télécharge celui de 2020 dans `data/raw/` et les
fichiers légers de 2025 sont dans `data/raw/mal-2025/` (`inspect_mal2025` décrit
leur structure).

`prepare_interactions` garde les animés du catalogue présents dans ces listes et
construit les matrices d'évaluation. Avec les données 2025, **4 749 animés** sur les
4 922 tout public du catalogue sont couverts (contre 3 814 avec celles de 2020) :

- **positif** : statut en cours, terminé ou en pause, sauf note de 1 à 4 ;
  « abandonné » compte comme vu (jamais recommandé) mais pas comme positif ;
- **échantillon** : 100 000 utilisateurs ayant au moins 10 positifs ; animés
  positifs chez au moins 5 utilisateurs d'entraînement ;
- **splits** : 90 000 utilisateurs d'entraînement, 5 000 de validation, 5 000 de
  test, absents de l'entraînement (comme un profil saisi dans la démo) ; 20 % de
  leurs positifs sont masqués et doivent être retrouvés ;
- les **animés adultes** (`is_adult`, 63 animés) sont exclus partout
  (`EXCLURE_ADULTES` dans [src/config.py](src/config.py)) ;
- les lignes sont triées avant le tirage : une même graine donne les mêmes splits
  sur toutes les machines.

## Évaluation

Code commun : [src/common/evaluation.py](src/common/evaluation.py). Pour chaque
utilisateur de validation ou de test, la méthode reçoit 80 % de sa liste, classe
tous les animés non vus, et on compare son top K aux 20 % masqués :

- **Recall@K** : hits ÷ min(K, nombre de cibles) ; **nDCG@K** : idem, pondéré par le rang ;
  **HitRate@K** : au moins un hit ;
- **couverture@10** : part du catalogue recommandée à au moins un utilisateur ;
  **popularité moyenne@10** : part des utilisateurs ayant vu les animés recommandés.

Le split `val` sert à régler les paramètres ; `test` est réservé aux chiffres finaux.
Les cibles sont tirées au hasard dans la liste (le dataset n'a pas de dates), ce qui
donne des scores élevés : seules les comparaisons entre méthodes ont du sens.

## Résultats

### Test final du modèle utilisé par la démo

Sur le catalogue MAL 2025 de 4 749 animés, EASE (λ = 500) et la popularité ont été
entraînés sur les mêmes 90 000 utilisateurs et évalués sur les mêmes 5 000 utilisateurs
de test. Leurs listes contenaient 263 139 titres masqués au total.

| Méthode | Recall@10 | nDCG@10 | HitRate@10 | Couverture@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularité | 0,2999 | 0,3300 | 0,8022 | 1,52 % |
| EASE | 0,7328 | 0,7625 | 0,9714 | 53,11 % |

EASE retrouve au moins un titre masqué parmi ses dix suggestions pour 97,14 % des
utilisateurs de test. L'évaluation utilise tout l'historique visible de chaque profil.
La démo accepte un profil MAL ou un à trois favoris saisis par le visiteur.
Les chiffres du tableau mesurent le classement EASE par titre MAL, avant le regroupement par série.

### Recommandation à partir d'un profil (split `val`, paramètres par défaut)

Tableau complet : [results/recommandation/resultats.csv](results/recommandation/resultats.csv)
(la source des données est dans la colonne `params`, champ `donnees`). Chaque série
a été exécutée en une fois : même split, 5 000 utilisateurs de validation, animés
adultes exclus.

**Listes MAL d'octobre 2025** (4 749 animés) :

| Méthode | Recall@10 | nDCG@10 | HitRate@10 | Couverture@10 | Popularité moy.@10 |
| --- | --- | --- | --- | --- | --- |
| popularity | 0,295 | 0,324 | 0,792 | 1,6 % | 0,560 |
| item_knn (k = 100, shrink = 10) | 0,499 | 0,538 | 0,941 | 34,9 % | 0,405 |
| **ease (λ = 500)** | **0,738** | **0,767** | **0,972** | **52,3 %** | 0,272 |
| synopsis_embeddings (bge-base) | 0,029 | 0,027 | 0,222 | 5,5 % | 0,079 |

**Entonnoir** (`funnel`, même split 2025) : l'étape 1 réunit 284 candidats par
utilisateur, qui contiennent 63 % des animés masqués ; LightGBM (18 caractéristiques)
les reclasse. Il apprend sur 5 000 utilisateurs d'entraînement réservés, que les
générateurs n'ont pas vus ; la référence est donc EASE entraîné sur les mêmes
85 000 utilisateurs.

| Méthode | Recall@10 | nDCG@10 | HitRate@10 | Couverture@10 | Popularité moy.@10 |
| --- | --- | --- | --- | --- | --- |
| ease (85 000 utilisateurs) | 0,737 | 0,766 | 0,972 | 52,1 % | 0,273 |
| **funnel** | **0,744** | **0,772** | **0,975** | 49,6 % | 0,293 |

Gain modeste (+0,006 de nDCG@10) : le re-ranker s'appuie surtout sur le rang EASE
(77 % du gain des arbres), puis sur la franchise déjà vue et la taille du profil.
Les autres caractéristiques n'apportent presque rien d'autre qu'EASE.

Listes de 2020 (3 814 animés), pour comparaison : nDCG@10 de 0,265 (popularity),
0,494 (item_knn), 0,727 (ease) et 0,032 (synopsis_embeddings). Le classement des
méthodes est identique ; les valeurs absolues ne se comparent pas d'un dataset à
l'autre (catalogue et utilisateurs différents).

Le filtrage collaboratif domine largement : les co-visionnages en disent bien plus
que le texte sur ce qu'un utilisateur va aimer. Le contenu seul recommande des animés
proches par le thème mais peu regardés ; il sert pour les animés sans interactions
(voir cold-start ci-dessous).

### Animés jamais regardés (cold-start)

`synopsis_finetune` affine bge-base par apprentissage contrastif sur ~15 000 paires
d'animés souvent regardés ensemble (hors franchise), puis évalue sur 751 animés mis
de côté, jamais vus pendant l'entraînement. Tableau :
[results/finetune/bge-base-en-v1.5.csv](results/finetune/bge-base-en-v1.5.csv).

| Méthode | Voisins retrouvés (Recall@10) | nDCG@10 | HitRate@10 |
| --- | --- | --- | --- |
| Hasard | — | 0,016 | 0,118 |
| bge-base | 0,076 | 0,042 | 0,239 |
| **bge-base affiné** | **0,127** | **0,059** | **0,336** |
| EASE (borne haute : a vu les interactions) | — | 0,604 | 0,950 |

### Cartes

[Carte des synopsis](results/embeddings/carte_bge-base-en-v1.5.html) et
[carte collaborative](results/embeddings/carte_collaboratif.html) (UMAP, HTML
autonome). Jaccard moyen des genres avec les 10 plus proches voisins : 0,36
(synopsis) et 0,42 (co-visionnages), contre 0,15 entre animés pris au hasard.

## Organisation

| Chemin | Contenu | Versionné |
| --- | --- | --- |
| `src/main.py` | Lanceur | oui |
| `src/config.py` | Chemins, clés API (lues dans `.env`), graine, filtre adulte | oui |
| `src/methods/` | Une méthode par fichier | oui |
| `src/preparation/` | Téléchargement et préparation des données | oui |
| `src/common/` | Code partagé : MinHash, chargement des splits, évaluation | oui |
| `scripts/` | `kaggle_run.py` (exécution sur Kaggle), `check_env.py` | oui |
| `tests/` | Tests unitaires (MinHash, lanceur, métriques) | oui |
| `docs/` | Plan, consignes, notes | oui |
| `data/content/` | Catalogue et fichiers `.train` dérivés | oui |
| `data/raw/`, `data/processed/` | Listes MAL téléchargées, matrices et splits | non |
| `models/` | Embeddings et modèles | non |
| `results/` | Paires MinHash, tableaux comparatifs, cartes, voisins | oui |

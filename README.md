# sjtu-data-mining-project

## Environnement

Python 3.12, sans venv :

```powershell
python -m pip install -r requirements.txt
copy .env.example .env          # puis renseigner MAL_CLIENT_ID, KAGGLE_USERNAME, KAGGLE_KEY
python scripts/check_env.py     # vérifie modules, GPU, catalogue et clés
```

`torch` s'installe par défaut en version CPU. Pour utiliser le GPU NVIDIA
(entraînement du Transformer, embeddings plus rapides), installer la version CUDA
(vérifier la commande exacte sur pytorch.org) :

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

## Lancer les méthodes

Tout passe par `src/main.py`, qui exécute les étapes et méthodes demandées, dans l'ordre :

```powershell
python src/main.py --list                                # étapes et méthodes disponibles
python src/main.py download_interactions prepare_interactions   # une fois : listes MAL -> matrices
python src/main.py popularity item_knn ease              # évaluer plusieurs méthodes
python src/main.py ease --lam 300 --split test           # options transmises aux méthodes
python src/main.py ease --help                           # options d'une méthode
python -B -m unittest discover -s tests -v               # tests
```

Les méthodes de recommandation sont évaluées sur le même protocole
(`src/common/evaluation.py`). Chaque exécution ajoute une ligne à
`results/recommandation/resultats.csv` et affiche les voisins de *Attack on Titan*
(`--exemple <mal_id>` pour un autre anime, `--exemple 0` pour aucun).

**Ajouter une méthode** : créer `src/methods/<nom>.py` contenant `DESCRIPTION`,
`ajouter_arguments(parser)` et `executer(args)`, puis l'enregistrer dans
`METHODES` (`src/main.py`). Pour une méthode de recommandation, `executer` entraîne
le modèle puis appelle `lancer_evaluation(nom, scorer, args, params)`, où
`scorer(X)` renvoie les scores des animes pour les utilisateurs de la matrice `X`.

## Organisation

| Chemin | Contenu | Versionné |
| --- | --- | --- |
| `src/main.py` | Lanceur | oui |
| `src/config.py` | Chemins, clés API (lues dans `.env`), graine | oui |
| `src/methods/` | Une méthode par fichier (MinHash, popularité, Item-KNN, EASE...) | oui |
| `src/preparation/` | Téléchargement et préparation des données | oui |
| `src/common/` | Code partagé : MinHash, chargement des splits, évaluation | oui |
| `scripts/check_env.py` | Vérification de l'environnement | oui |
| `tests/` | Tests unitaires | oui |
| `data/content/` | Catalogue de 4 985 animes et fichiers `.train` dérivés | oui |
| `data/raw/` | Données téléchargées (Kaggle, API) | non |
| `data/processed/` | Matrices utilisateurs x animes et splits d'évaluation | non |
| `models/` | Modèles entraînés, embeddings | non |
| `results/` | Paires MinHash, tableau comparatif des méthodes | oui |

## Données d'interactions

`prepare_interactions` lit `animelist.csv` (Kaggle *Anime Recommendation Database
2020* : 109 M lignes, 325 770 utilisateurs) et ne garde que les animes de notre
catalogue présents dans ces listes (4 017 sur 4 985 ; les plus récents sont absents).

- **Positif** : statut en cours, terminé ou en pause, sauf note de 1 à 4.
  « Abandonné » compte comme vu (jamais recommandé) mais pas comme positif ;
  « prévu » est ignoré.
- **Échantillon** : 100 000 utilisateurs ayant au moins 10 positifs ; animes
  positifs chez au moins 5 utilisateurs d'entraînement, soit 3 814 animes.
- **Splits** : 90 000 utilisateurs d'entraînement ; 5 000 de validation et 5 000
  de test, absents de l'entraînement (comme un profil MAL saisi dans la démo).
  Pour eux, 20 % des positifs sont masqués et doivent être retrouvés.
- Options : `--num-users`, `--min-positives`, `--val-users`, `--holdout`, `--seed`...

Premiers résultats (split `val`, paramètres par défaut, non réglés) :

| Méthode | Recall@10 | nDCG@10 | HitRate@10 | Couverture@10 | Popularité moy.@10 |
| --- | --- | --- | --- | --- | --- |
| popularity | 0,238 | 0,262 | 0,776 | 2,1 % | 0,501 |
| item_knn (k = 100, shrink = 10) | 0,455 | 0,491 | 0,942 | 36,5 % | 0,327 |
| ease (λ = 500) | **0,688** | **0,724** | **0,979** | **57,5 %** | **0,229** |

Recall@K est normalisé par min(K, nombre de cibles) ; les utilisateurs ont en
moyenne environ 44 cibles, donc Recall@10 vaut ici la précision@10. La popularité
moyenne est la part des utilisateurs d'entraînement ayant vu les animes recommandés
(plus bas = recommandations moins évidentes).

## MinHash sur les genres et tags

```powershell
python src/main.py prepare_content minhash_genres_tags
```

`prepare_content` crée notamment `data/content/anime_genres_tags.train` : trois
champs séparés par des tabulations, sans en-tête (`id`, `genres`, `tags`).
Ce fichier conserve les animes ayant au moins un genre ou tag, indépendamment
de leur synopsis : 4 985 animes dans le dataset actuel.

`minhash_genres_tags` compare uniquement ces métadonnées, en utilisant un label complet
par shingle (`k = 1`). Ainsi, `Slice of Life` reste une seule catégorie, l'ordre
des labels n'a pas d'effet et les doublons sont supprimés. La casse et les espaces
sont normalisés ; les préfixes `genre:` et `tag:` distinguent les deux champs.
Chaque label a le même poids. Les ensembles vides sont ignorés.

Les signatures utilisent 128 fonctions de hachage avec une graine fixe (42).
Le traitement des synopsis (`python src/main.py minhash_synopsis`) utilise les mêmes
128 fonctions : les paramètres et coefficients sont partagés via `src/common/minhash.py`.
Les synopsis conservent leurs shingles de trois mots.
Sur 20 000 paires tirées au hasard avec la graine 123, l'erreur absolue moyenne
par rapport au Jaccard exact est de 0,01376, contre 0,04096 avec 10 fonctions.
Cette mesure concerne ce dataset et cet échantillon.

Les paires dont le Jaccard **estimé est strictement supérieur à 0,5** sont écrites
dans `results/minhash/prediction_genres_tags.csv`, avec le Jaccard exact pour vérification
(colonnes `article1,article2,Est. J,Act. J`). Le seuil initial de 0,5 est configurable,
de même que le nombre de fonctions et la graine :

```powershell
python src/main.py minhash_genres_tags --num-hashes 256 --threshold 0.3 --seed 42
```

## MinHash sur les genres, tags et synopsis combinés

```powershell
python src/main.py minhash_genres_tags_synopsis
```

Ce troisième traitement lit `data/content/anime_genres_tags_synopsis.train`, généré
par `prepare_content` : quatre champs séparés par des tabulations, sans en-tête
(`id`, `genres`, `tags`, `synopsis`). Il utilise les mêmes 4 974 animes que les
synopsis seuls et les mêmes 128 fonctions de hachage (graine 42).

Chaque anime est représenté par l'union des labels complets de genres et tags
(`k = 1`) et des shingles de trois mots du synopsis (`k = 3`). Les préfixes
`genre:`, `tag:` et `synopsis:` distinguent les champs. Aucun shingle ne traverse
la frontière entre deux champs. Chaque shingle a le même poids : un synopsis
long contribue davantage de shingles que les métadonnées ; ce calcul ne donne
pas un poids égal aux trois champs.

Les paires dont le Jaccard estimé est strictement supérieur à 0,1 sont exportées
dans `results/minhash/prediction_genres_tags_synopsis.csv`, avec leur Jaccard exact calculé
sur l'ensemble combiné. Les options `--input`, `--output`, `--num-hashes`,
`--seed` et `--threshold` fonctionnent comme pour les genres/tags seuls.

- ajouter des fonctions de hashage
- normaliser les textes
- essayer d'autres valeurs de k (shingles)

Huit idées pour améliorer votre projet
Voici les pistes que je privilégierais, en tenant compte de votre code actuel, des méthodes enseignées et de ce qui pourrait être intéressant pour votre projet final.
Priorité	Idée	Méthodes du cours	Difficulté
1	Améliorer et évaluer MinHash	Shingling, MinHash, Jaccard	Facile
2	Ajouter TF-IDF et comparer les méthodes	Similarité cosinus	Facile
3	Construire un recommandateur hybride	Similarité, clustering	Moyenne
4	Accélérer la recherche avec LSH	Locality-Sensitive Hashing	Moyenne
5	Détecter des communautés d'animes	Graphes, community detection	Moyenne
6	Visualiser une carte des animes	Clustering, réduction de dimension	Moyenne
7	Classer les animes par importance	PageRank, link analysis	Moyenne
8	Étudier l'évolution des tendances	Streaming algorithms	Avancée

Idée 1 — Améliorer votre MinHash actuel
Commencez par les améliorations déjà évoquées dans votre README : normalisation des textes, fonctions de hashage et différentes tailles de shingles.

Vous pourriez comparer les configurations suivantes :
Exemples de configurations à tester
Configuration	Shingles k	Hash functions
Configuration actuelle	3	10
Expérience A	2	100
Expérience B	3	100
Expérience C	3	200
Expérience D	1	100

Configurations expérimentales proposées, et non résultats déjà obtenus.
Mesurez l'erreur moyenne entre les similarités estimées et réelles, ainsi que les temps de calcul. Augmenter le nombre de fonctions ne garantit pas une meilleure estimation pour chaque paire, mais réduit généralement la variance de l'estimateur.

Attention à un problème dans les données : j'ai identifié 11 synopsis contenant moins de trois mots. Ils ne produisent aucun 3-shingle, ce qui donne des signatures MinHash artificiellement identiques avec votre implémentation actuelle. Il faudra filtrer ou traiter ces cas à part.
Idée 2 — Comparer MinHash, TF-IDF et similarité cosinus
C'est probablement l'amélioration offrant le meilleur rapport effort/résultat.

Au lieu de mesurer uniquement des shingles identiques, TF-IDF représente chaque synopsis sous forme d'un vecteur pondérant les mots et expressions.

J'ai testé rapidement cette approche sur vos 4 985 animes, avec des unigrammes et bigrammes.









Exemple : Attack on Titan

Résultats obtenus avec un premier modèle TF-IDF


Anime rapproché	Cosinus
Attack on Titan: Junior High	0,233
Attack on Titan Season 2	0,220
Attack on Titan Season 3 Part 2	0,211
Attack on Titan: Final Season Part 2	0,209

Résultats expérimentaux réels calculés sur le dataset fourni. Ils montrent une capacité à rapprocher les œuvres d'une même franchise, mais pas encore une recommandation optimale.
Le fait que Junior High arrive avant la saison 2 illustre aussi une limite intéressante : des textes proches ne correspondent pas toujours au même type d'expérience pour le spectateur.

Vous pourriez ensuite comparer TF-IDF, MinHash et, en extension, des embeddings de phrases avec Sentence Transformers.
Idée 3 — Créer un système de recommandation hybride
C'est l'application la plus naturelle de votre dataset, qui contient déjà les genres, tags, scores et studios.

Vous pourriez calculer un score combinant plusieurs dimensions :
\[
S(A,B)=0.6S_{\text{text}}+0.25S_{\text{genres}}+0.15S_{\text{tags}}
\]

Les poids sont des hypothèses initiales à ajuster expérimentalement.

Exemple : un utilisateur recherche des animes similaires à Death Note. Le système analyse non seulement son synopsis, mais aussi ses genres et tags, puis propose des animes apparentés tout en pouvant exclure les suites, remakes ou épisodes spéciaux.

L'amélioration intéressante serait d'offrir deux modes : « Même univers / même franchise » et « Découvrir de nouveaux animes similaires ».
Idée 4 — Ajouter LSH pour accélérer la recherche
Votre programme compare actuellement toutes les paires, soit environ 12,4 millions de comparaisons.

Le Locality-Sensitive Hashing permettrait de sélectionner des candidats probables sans toutes les comparer.

Vous pourriez montrer le compromis entre la vitesse, la mémoire et le taux de paires pertinentes retrouvées. Cela correspond très directement à la Lecture 3.
Idée 5 — Détecter des communautés d'animes









Transformez vos données en graphe dans lequel chaque nœud représente un anime et chaque arête relie deux animes dont la similarité dépasse un seuil.

Utilisez ensuite Louvain ou Girvan-Newman pour détecter des communautés.

Vous pourriez découvrir des regroupements autour de la fantasy, du sport, de la romance ou de la science-fiction, puis examiner les communautés qui mélangent plusieurs genres.

Cela correspond à la Lecture 6, Community Detection. Une communauté détectée n'est toutefois pas nécessairement un genre : il faudra interpréter les résultats grâce aux métadonnées.
Idée 6 — Créer une carte interactive des animes
Utilisez PCA ou UMAP sur les vecteurs TF-IDF pour projeter les animes en deux dimensions.

Chaque point serait un anime, avec une couleur représentant son genre principal ou sa communauté.

On pourrait alors sélectionner un anime et explorer les titres proches. C'est un excellent complément visuel au rapport final.
Idée 7 — Utiliser PageRank pour améliorer les recommandations
Sur votre graphe d'animes similaires, calculez PageRank pour identifier les titres centraux.

Comparez ensuite cette centralité avec les notes, les favoris et le nombre de membres déjà présents dans le dataset.

Une question de recherche possible serait : les animes les plus centraux dans un réseau de similarité sont-ils aussi les plus populaires ?

Attention : PageRank mesure ici la centralité dans le graphe construit, pas directement la qualité d'un anime.
Idée 8 — Analyser les tendances au fil du temps
Votre dataset dispose de dates de sortie. Vous pouvez simuler l'arrivée progressive de nouveaux animes, année après année, et observer comment les genres dominants changent.

Cela permettrait d'expérimenter les fenêtres glissantes et l'échantillonnage vus en Lecture 4, Mining Data Streams.

Ce serait un flux historique simulé, et non une véritable collecte en temps réel.

3. Le projet final que je vous recommanderais
AnimeScope — Anime Similarity & Recommendation Explorer
Recherche, recommandation et découverte de communautés d'animes à partir des techniques de Data Mining

Je construirais le projet autour de quatre modules complémentaires :
Dataset MyAnimeList — 4 985 animes
Synopsis, genres, tags, studios, notes

Moteur de similarité
MinHash + LSH + TF-IDF

Recommandateur hybride
Texte + genres + tags

Analyse de communautés
Graphes + clustering

Évaluation
Qualité + vitesse + mémoire



Interface interactive avec Streamlit
Recherche d'animes, Top 5 recommandations, explications et carte des communautés

4. Comment évaluer réellement le projet ?
C'est un aspect que je vous encourage particulièrement à travailler. Une interface de recommandation ne suffit pas : vous devez pouvoir démontrer que vos méthodes apportent une amélioration.
Question	Mesure proposée
MinHash est-il précis ?	MAE entre Jaccard estimé et réel
LSH accélère-t-il la recherche ?	Temps, mémoire, rappel des paires similaires
Quel modèle recommande le mieux ?	Precision@5, nDCG@5
Les recommandations sont-elles variées ?	Diversité des genres proposés
Les communautés sont-elles cohérentes ?	Modularité et analyse des genres

Pour évaluer la pertinence des recommandations, votre équipe pourrait annoter manuellement les cinq premières suggestions de 30 animes différents. Les genres ne suffisent pas, à eux seuls, à fournir une vérité terrain fiable.
5. Répartition du travail entre trois étudiants
La présentation du cours indique que le projet ouvert compte pour 25 % de la note finale et se réalise par groupes de trois. Je répartirais donc les responsabilités ainsi :
Étudiant	Responsabilité principale	Livrables
1	Amélioration de MinHash, implémentation LSH, benchmarking	Comparatifs précision/vitesse
2	TF-IDF et recommandation hybride	Moteur Top 5 et métriques
3	Graphes, community detection et interface	Visualisations et démonstration

Les trois étudiants participeraient ensuite ensemble à l'évaluation des recommandations et au rapport.
6. Ce que je ferais en premier
Je commencerais par fiabiliser les données et le code : traiter les synopsis vides ou trop courts, corriger l'absence de shingles, vérifier la reproductibilité de prediction.csv et rendre les paramètres configurables.

Ensuite, je construirais un véritable benchmark MinHash contre TF-IDF, avant de développer le recommandateur hybride.

Mon choix serait de concentrer le projet sur les idées 1 à 5, avec la carte interactive comme fonctionnalité supplémentaire si vous avez suffisamment de temps. Vous pourriez ainsi couvrir trois chapitres importants du cours — similarité, analyse de liens et détection de communautés — sans transformer le projet en simple application web.

Le résultat serait à la fois une démonstration concrète utilisable, un projet techniquement intéressant et une expérimentation rigoureuse, ce qui me semble particulièrement adapté à AI3602.

# Méthodes MinHash

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

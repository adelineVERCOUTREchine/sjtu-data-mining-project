# sjtu-data-mining-project

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
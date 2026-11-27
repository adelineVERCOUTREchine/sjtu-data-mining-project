# Méthodes de recommandation d'animes : état de l'art et plan

Ce document recense les méthodes les plus adaptées à notre objectif. Il indique aussi les données à ajouter pour les mettre en œuvre.

**Objectif.** Recommander des animes dans deux situations :

1. **À partir d'un anime choisi** (*item-to-item*) : « J'ai aimé *Death Note*, que regarder ensuite ? »
2. **À partir d'un profil MyAnimeList** (*user-to-item*) : l'utilisateur saisit son pseudo MAL. Le système récupère sa liste (notes, statuts) et propose des animes qu'il n'a pas encore vus.

---

## 1. Point de départ : ce que nous avons déjà

| Élément | État actuel |
| --- | --- |
| Dataset | `data/content/anime_dataset.csv` : 4 985 animes populaires (≥ 25 810 membres MAL), avec `mal_id`, titres, format, source, année, score, membres, favoris, genres, tags, studios et synopsis |
| Similarité | MinHash avec 128 fonctions de hachage. La similarité de Jaccard est calculée sur les genres et tags (`k = 1`), les synopsis (`k = 3`) et les deux combinés |
| Sortie | Paires d'animes au-dessus d'un seuil de Jaccard (`results/minhash/prediction_*.csv`) |

**Limite principale.** Notre dataset contient uniquement des **attributs de contenu**. Il ne contient **aucune interaction utilisateur–anime**. Or les meilleures méthodes de recommandation reposent sur le **filtrage collaboratif** : « les utilisateurs qui ont aimé A ont aussi aimé B ». Il nous faut donc un dataset de listes MAL (section 2).

Notre `mal_id` permet une **jointure directe** avec tous les datasets MAL ci-dessous. Nos métadonnées restent utiles pour la partie contenu, le cold-start et les explications.

---

## 2. Données à ajouter

### 2.1. Interactions utilisateurs (indispensable pour le filtrage collaboratif)

| Dataset | Contenu (ordres de grandeur, à vérifier au téléchargement) | Intérêt |
| --- | --- | --- |
| **Kaggle — *Anime Recommendation Database 2020*** (hernan4444) | ≈ 17 500 animes, ≈ 325 000 utilisateurs, ≈ 109 M lignes dans `animelist.csv` (`user_id, anime_id, score, watching_status, watched_episodes`), plus `rating_complete.csv` (animes terminés et notés) | **Recommandé.** Gros volume, statuts de visionnage (vu, abandonné, en cours, prévu), IDs MAL |
| Kaggle — *Anime Dataset 2023* (dbdmobile) | `users-score-2023.csv` (≈ 24 M notes), `users-details-2023.csv` | Plus récent, mais seulement des notes (pas de statut) |
| Kaggle — *MyAnimeList Dataset* (azathoth42, 2018) | `animelists_cleaned.csv` avec **dates de mise à jour** | Utile si l'on veut des modèles séquentiels ou un découpage temporel |
| Kaggle — *Anime Recommendations Database* (CooperUnion, 2016) | ≈ 73 000 utilisateurs, ≈ 7,8 M notes | Ancien, mais léger : bon pour prototyper |

**Conseils pratiques.**
- Les fichiers font plusieurs Go. Il faut lire par blocs (pandas `chunksize`, polars ou DuckDB) et **sous-échantillonner** : environ 50 000 à 100 000 utilisateurs actifs suffisent pour un projet de cours.
- Il faut **restreindre le catalogue** aux animes d'un minimum de popularité. Nos 4 985 animes font un bon catalogue, car ils sont déjà filtrés sur la popularité.
- Il faut convertir les listes en **feedback implicite**. On considère « vu » (statut *completed* ou *watching*) comme positif. La note (1–10) sert de poids de confiance. Le statut *dropped* compte comme négatif faible, et *plan to watch* comme intérêt faible ou à ignorer.

### 2.2. Profil d'un utilisateur réel (pour la démo)

- **API officielle MyAnimeList v2** : `GET https://api.myanimelist.net/v2/users/{user_name}/animelist?fields=list_status&limit=1000`, avec l'en-tête `X-MAL-CLIENT-ID`. Un Client ID gratuit s'obtient dans les paramètres API du compte MAL. Cet endpoint renvoie les listes publiques : note, statut, épisodes vus et date de mise à jour.
- Jikan (API non officielle) a retiré la lecture des listes utilisateur dans sa v4. Il faut donc privilégier l'API officielle.
- Pour une démo sans réseau, il faut prévoir un **export XML MAL** (fonction « Export » du site) ou quelques profils en cache.

### 2.3. Relations et vérité terrain (pour l'item-to-item)

- **Relations entre œuvres** (suite, préquelle, spin-off, film récapitulatif). On les récupère via l'API AniList GraphQL (champ `relations`) ou via Jikan `/anime/{id}/relations`. Notre dataset vient apparemment d'AniList (champs `title_romaji`, `cover_image`). Ces relations permettent de **regrouper les franchises**. Sans ce regroupement, les recommandations sont saturées de suites : c'est déjà le cas avec TF-IDF sur *Attack on Titan*.
- **Recommandations votées par les utilisateurs** : Jikan `/anime/{id}/recommendations` (paires « si vous avez aimé X, regardez Y » proposées par les membres MAL) et le champ `recommendations` d'AniList. C'est une **vérité terrain humaine** pour évaluer les méthodes item-to-item.
- Données facultatives : staff (réalisateur, compositeur) et rang de pertinence des tags AniList (`tags { name rank }`). Ce rang permet de pondérer les tags au lieu de leur donner le même poids.

---

## 3. Méthodes pour la recommandation à partir d'un anime (item-to-item)

### 3.1. Similarité de contenu (déjà commencée)

| Méthode | Principe | Avis |
| --- | --- | --- |
| MinHash + Jaccard (actuel) | Jaccard estimé sur des ensembles de labels ou de shingles | Bonne base pédagogique. Les 3-shingles de synopsis capturent surtout des formulations identiques, donc surtout les franchises |
| **MinHash LSH** (banding) | On découpe la signature en `b` bandes de `r` lignes, et seules les paires qui partagent une bande deviennent candidates | Évite les ≈ 12,4 M comparaisons. On peut tracer la courbe en S en faisant varier `b` et `r`. Cela correspond au cours |
| Jaccard **pondéré** sur genres et tags | Poids IDF pour chaque label (un tag rare comme « Time Loop » compte plus que « Action »), plus le rang AniList | Gain important pour un coût faible |
| TF-IDF / BM25 + cosinus | Vecteurs de mots pondérés pour le synopsis | Meilleur que les 3-shingles pour le sens global |
| **Embeddings de phrases** | On encode le synopsis avec un modèle pré-entraîné (`all-MiniLM-L6-v2`, `bge-m3`, `multilingual-e5`) puis on calcule le cosinus | Méthode la plus forte pour le contenu textuel, car elle capture des thèmes proches exprimés avec d'autres mots. Le calcul est instantané pour 5 000 animes |
| SimHash / hyperplans aléatoires | LSH pour la similarité cosinus sur des vecteurs denses | Équivalent de MinHash LSH pour les embeddings. On peut aussi utiliser FAISS |

**Contenu hybride.** On combine les scores par une somme pondérée :

```
S(A,B) = w1·cos(emb_A, emb_B) + w2·Jaccard_IDF(tags) + w3·Jaccard(genres) + w4·[même studio] + w5·proximité(année)
```

Les poids sont appris ou réglés sur la vérité terrain de la section 2.3.

### 3.2. Similarité collaborative (nécessite la section 2.1)

C'est généralement **bien meilleur que le contenu** pour les animes populaires. Deux animes sont proches si les mêmes personnes les ont aimés.

| Méthode | Principe | Avis |
| --- | --- | --- |
| **Item-KNN cosinus** | Cosinus entre les colonnes de la matrice utilisateurs × animes | Simple et explicable (« 72 % des fans de X ont vu Y ») |
| Lift / PMI | `P(A,B) / (P(A)·P(B))` | Corrige le biais vers les animes très populaires |
| **EASE^R** (Steck, 2019) | Matrice item-item `B` obtenue en forme fermée : `B = I − P·diag(1/diag(P))`, avec `P = (XᵀX + λI)⁻¹` | **Excellent rapport qualité/effort.** Une seule inversion de matrice d'environ 5 000 × 5 000. Souvent parmi les meilleurs résultats publiés sur les benchmarks |
| Embeddings ALS / BPR | Vecteurs latents des animes issus de la factorisation (section 4), comparés par cosinus | Donne aussi une carte 2D (UMAP) des animes |

### 3.3. Graphe (lien direct avec le chapitre « Graph Mining »)

- On construit un **graphe biparti** utilisateurs–animes.
- **P³α / RP³β** (Christoffel et al., Paudel et al.) : marche aléatoire de 3 pas, anime → utilisateur → anime. RP³β pénalise la popularité par `pop^β`. C'est très simple, et c'est compétitif avec des modèles bien plus lourds.
- **Personalized PageRank** (random walk with restart) depuis l'anime choisi.
- Pour l'analyse : détection de communautés (Louvain) sur le graphe anime–anime. On interprète ensuite les communautés avec les genres et tags.

### 3.4. Post-traitement indispensable

- **Regroupement des franchises** : grâce aux relations de la section 2.3, on garde au plus une entrée par franchise, ou on propose deux modes (« même univers » et « découvrir autre chose »).
- **Diversité (MMR)** : on re-classe avec `λ·pertinence − (1−λ)·max_sim(déjà choisis)` pour éviter 10 recommandations quasi identiques.
- Filtres optionnels sur le format (exclure CM, PV, Music) et le contenu adulte.

---

## 4. Méthodes pour la recommandation à partir d'un profil MAL (user-to-item)

### 4.1. Principe : le « fold-in » d'un nouvel utilisateur

L'utilisateur de la démo **n'est pas dans le dataset d'entraînement**. Il faut donc des méthodes capables de l'intégrer **sans réentraînement** :

- **EASE^R** : le score est `x_u · B`, où `x_u` est le vecteur de sa liste MAL. L'intégration est **immédiate**, ce qui est idéal.
- **Item-KNN / RP³β** : on agrège les similarités des animes qu'il a aimés, pondérées par ses notes.
- **ALS implicite** : on résout un petit problème de moindres carrés pour son vecteur `u` en gardant les facteurs des animes fixes (la bibliothèque `implicit` le fait avec `recalculate_user=True`).
- **Contenu** : le profil est la moyenne des embeddings des animes vus, pondérée par la note centrée (`note − moyenne de l'utilisateur`). Les animes mal notés contribuent négativement.

### 4.2. Modèles, du plus simple au plus avancé

| Rang | Méthode | Type | Pourquoi |
| --- | --- | --- | --- |
| 0 | Popularité, et popularité par genre préféré | Baseline | **Obligatoire** pour montrer que les autres méthodes apportent quelque chose |
| 1 | Profil de contenu (embeddings + tags pondérés) | Contenu | Fonctionne même pour les animes récents sans interactions (cold-start item) |
| 2 | Item-KNN, RP³β | Voisinage / graphe | Simples, explicables et solides |
| 3 | **EASE^R** | Linéaire item-item | Probablement **le meilleur modèle simple**. Fold-in trivial |
| 4 | **ALS implicite** (Hu, Koren, Volinsky, 2008) | Factorisation | Standard industriel. La confiance vaut `1 + α·note`. Bibliothèque `implicit` |
| 5 | BPR / Mult-VAE | Factorisation / neuronal | Pour comparer avec ALS |
| 6 | **LightGCN** (He et al., 2020) | GNN sur le graphe biparti | État de l'art « classique ». S'inscrit bien dans le chapitre graphes. Plus long à entraîner (RecBole, PyTorch) |
| 7 | SASRec / BERT4Rec | Séquentiel (Transformer) | Seulement si l'on a des dates (dataset 2018). Exploite l'ordre de visionnage |

### 4.3. Architecture recommandée : un entonnoir en deux étages

C'est l'architecture suggérée par les consignes du cours (« funnel-based retrieval and re-ranking ») :

```
Liste MAL de l'utilisateur
        │
        ▼
┌─────────────── Étape 1 : Retrieval (≈ 200 candidats) ───────────────┐
│  EASE^R  ∪  ALS  ∪  profil de contenu (embeddings)  ∪  popularité    │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────── Étape 2 : Re-ranking ─────────────────────────────────┐
│  LightGBM LambdaMART avec comme features : scores de chaque          │
│  retriever, score MAL, popularité, année, format, affinité de genre  │
│  de l'utilisateur, appartenance à une franchise déjà vue, etc.       │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
Post-traitement : franchises (suite d'un anime vu = « à continuer »),
diversité MMR, explications (« car vous avez aimé X et Y »)
```

À défaut de re-ranker appris, une **fusion de rangs** (*Reciprocal Rank Fusion* : `Σ 1/(60 + rang)`) des retrievers est un bon hybride sans entraînement.

### 4.4. Extension avec un LLM (thème « LLM et data mining » du cours)

- **Re-ranking par LLM** : on donne au modèle les 30 meilleurs candidats, avec leurs synopsis et tags, et un résumé du profil (animes les mieux notés, genres préférés). Le modèle classe les 10 meilleurs et **génère une explication**.
- **Requêtes en langage naturel** : « un anime court, sombre, avec un twist, comme *Erased* ». On combine des embeddings de la requête et des filtres sur les métadonnées.
- **LLM-as-a-Judge** : un LLM note la pertinence des listes recommandées. Ce n'est qu'un complément à l'évaluation hors ligne, pas un remplacement. À documenter dans le rapport d'usage de l'IA.

---

## 5. Évaluation

Une démo ne suffit pas : il faut **montrer chiffres à l'appui** quelle méthode est la meilleure. Le critère « Technical Correctness » compte pour 10 % de la note.

### 5.1. User-to-item (hors ligne)

- **Découpage** : pour chaque utilisateur test, on masque 20 % de ses animes aimés (aléatoirement, ou les plus récents si l'on a les dates). Le modèle doit les retrouver. Les utilisateurs test sont **absents de l'entraînement**, pour simuler le fold-in de la démo.
- **Métriques** : Recall@10, nDCG@10, MAP@10, HitRate@10.
- **Au-delà de la précision** : couverture du catalogue, nouveauté (popularité moyenne des recommandations), diversité intra-liste (distance moyenne des genres et embeddings), part des suites.
- On compare **toutes les méthodes contre la baseline popularité**.

### 5.2. Item-to-item

- **Vérité terrain** : les recommandations votées par les utilisateurs MAL ou AniList (section 2.3), **hors même franchise**.
- **Métriques** : Recall@10 et nDCG@10 par anime requête.
- **Comparaison** : MinHash (actuel), Jaccard IDF, TF-IDF, embeddings, contenu hybride, Item-KNN, EASE et RP³β.
- **Pour LSH** : rappel des vraies paires proches en fonction de `b` et `r`, temps de calcul et mémoire, comparés à la recherche exhaustive.

### 5.3. Évaluation humaine (petite et ciblée)

- On choisit environ 20 animes requêtes et environ 5 profils réels de l'équipe ou d'amis.
- Chaque membre note les 5 premières suggestions de chaque méthode **à l'aveugle** (méthode cachée).
- C'est utile pour le poster : un tableau avec les résultats « humains » et les résultats « hors ligne ».

---

## 6. Recommandation pour notre projet (calendrier serré)

La **présentation a lieu le 1er décembre** et le **rendu le 6 décembre**. Il faut donc privilégier les méthodes au meilleur rapport valeur/effort.

| Priorité | Tâche | Effort estimé |
| --- | --- | --- |
| P0 | Télécharger *Anime Recommendation Database 2020*, filtrer sur nos 4 985 `mal_id`, sous-échantillonner les utilisateurs et construire la matrice creuse | 0,5 jour |
| P0 | Baselines popularité + **EASE^R** + Item-KNN, avec évaluation Recall/nDCG@10 | 0,5 jour |
| P0 | Embeddings de synopsis + Jaccard IDF des tags → contenu hybride, comparé à notre MinHash | 0,5 jour |
| P1 | ALS implicite (`implicit`) et RP³β, ajoutés au tableau comparatif | 0,5 jour |
| P1 | Récupération d'un profil via l'API MAL v2 + démo **Streamlit** (pseudo MAL ou anime → top 10 avec explications) | 1 jour |
| P1 | Regroupement des franchises (relations AniList) + MMR | 0,5 jour |
| P2 | MinHash LSH : courbe rappel/temps (lien avec le cours) | 0,5 jour |
| P2 | Hybride par fusion de rangs, ou re-ranker LightGBM | 0,5 jour |
| P3 | LightGCN, re-ranking et explications par LLM | Si le temps le permet |

**Message clé pour le poster.** Le filtrage collaboratif (EASE, ALS) domine pour les utilisateurs ayant un historique. Le contenu (embeddings et tags) est indispensable pour le cold-start et les explications. Un **hybride**, avec un traitement des franchises, donne les recommandations les plus utiles. Nos chiffres hors ligne doivent montrer tout cela.

### Dépendances Python à prévoir

`pandas` (absent de l'environnement actuel) ou `polars`, `scipy`, `numpy`, `scikit-learn`, `implicit`, `sentence-transformers`, `lightgbm`, `requests`, `streamlit`, `umap-learn` (facultatif) et `faiss-cpu` (facultatif).

---

## 7. Références

- Steck, H. (2019). *Embarrassingly Shallow Autoencoders for Sparse Data* (EASE^R). WWW.
- Hu, Y., Koren, Y., Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
- Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI.
- Paudel, B. et al. (2017). *Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications* (RP³β). ACM TiiS.
- Liang, D. et al. (2018). *Variational Autoencoders for Collaborative Filtering* (Mult-VAE). WWW.
- He, X. et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR.
- Kang, W.-C., McAuley, J. (2018). *Self-Attentive Sequential Recommendation* (SASRec). ICDM.
- Dacrema, M. F. et al. (2019). *Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches*. RecSys. Cet article montre que des baselines simples bien réglées (ItemKNN, RP³β, EASE) battent souvent des modèles neuronaux : c'est un bon argument pour notre choix.
- Leskovec, Rajaraman, Ullman. *Mining of Massive Datasets*, ch. 3 (similarité, LSH), ch. 5 (PageRank) et ch. 9 (systèmes de recommandation).

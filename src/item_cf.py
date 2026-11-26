"""
Item-Based Collaborative Filtering sur rating.csv (user_id, anime_id, rating)
Sortie : similarities.csv avec les colonnes anime1, anime2, similarity_score

Étapes :
 1. Charger et filtrer (on retire les -1, les animes / users trop peu actifs)
 2. Centrer les notes par utilisateur  -> "aimé" = au-dessus de SA moyenne
 3. Similarité cosinus entre animes (adjusted cosine) + shrinkage
 4. Garder les K plus proches voisins de chaque anime
 5. Exporter les paires en CSV
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

RATING_PATH = "../archive/rating.csv"
OUTPUT_PATH = "../data/users.csv"

MIN_RATERS = 50       # un anime doit avoir au moins N notes pour être gardé
MIN_USER_RATINGS = 5  # un user doit avoir noté au moins N animes
K = 50                # nb de voisins gardés par anime
SHRINK = 20           # plus grand = plus de méfiance envers les paires peu observées
CHUNK = 500


def load(path):
    df = pd.read_csv(path)
    df = df[df.rating != -1]  # "vu mais pas noté" : ignoré

    cnt_item = df.anime_id.value_counts()
    df = df[df.anime_id.isin(cnt_item[cnt_item >= MIN_RATERS].index)]
    cnt_user = df.user_id.value_counts()
    df = df[df.user_id.isin(cnt_user[cnt_user >= MIN_USER_RATINGS].index)]

    df = df.copy()
    df["centered"] = df.rating - df.groupby("user_id").rating.transform("mean")
    return df


def build_matrices(df):
    users = df.user_id.astype("category")
    items = df.anime_id.astype("category")
    u, i = users.cat.codes.values, items.cat.codes.values
    shape = (users.cat.categories.size, items.cat.categories.size)

    X = sp.csr_matrix((df.centered.values.astype(np.float32), (u, i)), shape=shape)
    B = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (u, i)), shape=shape)
    return X, B, items.cat.categories.values


def item_similarity(X, B, k=K, shrink=SHRINK):
    """Matrice creuse S (items x items) avec les k meilleurs voisins par item."""
    Xn = normalize(X, norm="l2", axis=0).tocsc()
    Xt, Bt = Xn.T.tocsr(), B.T.tocsr()
    n = X.shape[1]
    k = min(k, n - 1)
    rows, cols, vals = [], [], []

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        sim = (Xt[start:end] @ Xn).toarray()
        co = (Bt[start:end] @ B).toarray()
        sim *= co / (co + shrink)

        for r in range(end - start):
            sim[r, start + r] = 0  # pas d'auto-similarité

        top = np.argpartition(-sim, k, axis=1)[:, :k]
        for r in range(end - start):
            s = sim[r, top[r]]
            keep = s > 0
            rows.extend([start + r] * int(keep.sum()))
            cols.extend(top[r][keep])
            vals.extend(s[keep])

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def export_csv(S, items, path):
    coo = S.tocoo()
    out = pd.DataFrame({
        "anime1": items[coo.row],
        "anime2": items[coo.col],
        "similarity_score": coo.data,
    })
    out = out.sort_values(["anime1", "similarity_score"], ascending=[True, False])
    out.to_csv(path, index=False)
    return out


if __name__ == "__main__":
    df = load(RATING_PATH)
    X, B, items = build_matrices(df)
    S = item_similarity(X, B)
    out = export_csv(S, items, OUTPUT_PATH)
    print(f"{len(out)} paires écrites dans {OUTPUT_PATH}")

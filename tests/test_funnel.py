import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from methods.funnel import CARACTERISTIQUES, Entonnoir, donnees_ranker, separer_profils


def petit_entonnoir(n=12, par_generateur=3, seed=0):
    """Entonnoir sur un catalogue synthétique, sans charger les vraies données."""
    rng = np.random.default_rng(seed)
    e = object.__new__(Entonnoir)
    e.n, e.par_generateur = n, par_generateur
    e.B = rng.normal(size=(n, n)).astype(np.float32)
    e.S_knn = rng.random((n, n)).astype(np.float32)
    e.S_contenu = rng.random((n, n)).astype(np.float32)
    e.popularite = rng.integers(1, 100, n).astype(np.float32)
    e.franchise = np.zeros((n, n), dtype=np.float32)
    e.franchise[0, 1] = e.franchise[1, 0] = 1.0
    e.note = rng.random(n).astype(np.float32) * 10
    e.annee = rng.integers(1990, 2025, n).astype(np.float32)
    e.episodes = np.log1p(rng.integers(1, 50, n)).astype(np.float32)
    e.format = rng.integers(0, 3, n).astype(np.float32)
    G = rng.integers(0, 2, (n, 5)).astype(np.float32) + 1e-3
    e.genres = G / np.linalg.norm(G, axis=1, keepdims=True)
    return e


class FunnelTests(unittest.TestCase):
    def test_candidats_excluent_les_animes_vus(self):
        e = petit_entonnoir()
        X = sp.csr_matrix(np.array([[1, 0, 0, 1] + [0] * 8, [0] * 11 + [1]], dtype=np.float32))
        X, scores, rangs, masque = e.candidats(X)
        self.assertFalse(masque[0, 0] or masque[0, 3] or masque[1, 11])
        # Union de 4 générateurs x 3 candidats : entre 3 et 12 candidats.
        self.assertTrue(all(3 <= k <= 12 for k in masque.sum(axis=1)))

    def test_caracteristiques_et_franchise(self):
        e = petit_entonnoir()
        X = sp.csr_matrix(np.array([[1] + [0] * 11], dtype=np.float32))
        X, scores, rangs, masque = e.candidats(X)
        masque[0, 1] = True  # forcer la suite de l'animé 0 parmi les candidats
        u, c, F = e.caracteristiques(X, scores, rangs, masque)
        self.assertEqual(F.shape, (masque.sum(), len(CARACTERISTIQUES)))
        self.assertTrue(np.isfinite(F).all())
        franchise = F[list(c).index(1), CARACTERISTIQUES.index("franchise_vue")]
        self.assertEqual(franchise, 1.0)

    def test_donnees_ranker_et_lightgbm(self):
        import lightgbm as lgb
        rng = np.random.default_rng(1)
        e = petit_entonnoir(par_generateur=6)
        profils = sp.csr_matrix((rng.random((30, 12)) < 0.5).astype(np.float32))
        entree, cible = separer_profils(profils, 0.3, rng)
        self.assertEqual((entree.multiply(cible)).nnz, 0)  # aucune cible dans l'entrée
        F, y, groupes = donnees_ranker(e, entree, cible, batch_size=8)
        self.assertEqual(len(F), len(y))
        self.assertEqual(groupes.sum(), len(y))
        modele = lgb.LGBMRanker(objective="lambdarank", n_estimators=5, min_child_samples=1,
                                verbose=-1).fit(F, y, group=groupes)
        self.assertEqual(modele.predict(F).shape, (len(y),))


if __name__ == "__main__":
    unittest.main()

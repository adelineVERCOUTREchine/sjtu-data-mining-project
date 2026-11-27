import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.donnees import Split
from common.evaluation import evaluer, top_k
from methods.synopsis_finetune import jetons_titre


def matrice(lignes, n_items=6):
    """Matrice binaire utilisateurs x animes à partir de listes d'indices."""
    rows = [u for u, items in enumerate(lignes) for _ in items]
    cols = [i for items in lignes for i in items]
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lignes), n_items))


class EvaluationTests(unittest.TestCase):
    def test_top_k_trie_par_score_decroissant(self):
        scores = np.array([[0.1, 0.9, 0.5, 0.7]])
        self.assertEqual(top_k(scores, 3).tolist(), [[1, 3, 2]])

    def test_metriques_sur_un_cas_calcule_a_la_main(self):
        # Utilisateur 0 : cibles {0, 3}, déjà vu {5}. Le modèle classe 5 > 0 > 1 > 3 > 2 > 4.
        split = Split("val", input=matrice([[5]]), target=matrice([[0, 3]]),
                      exclude=matrice([[5]]))
        scores = np.array([[0.9, 0.8, 0.1, 0.7, 0.0, 1.0]])
        popularite = np.linspace(0.1, 0.6, 6)
        r = evaluer(lambda X: scores, split, popularite)
        # Après exclusion de 5 : top = 0, 1, 3, 2, 4 -> hits aux rangs 1 et 3.
        self.assertAlmostEqual(r["recall@10"], 1.0)
        dcg = 1 + 1 / np.log2(4)
        idcg = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(r["ndcg@10"], dcg / idcg, places=6)
        self.assertEqual(r["hitrate@10"], 1.0)
        self.assertEqual(r["utilisateurs"], 1)

    def test_animes_deja_vus_jamais_recommandes(self):
        split = Split("val", input=matrice([[0]]), target=matrice([[1]]),
                      exclude=matrice([[0, 2]]))
        scores = np.array([[10.0, 1.0, 9.0, 0.0, 0.0, 0.0]])
        r = evaluer(lambda X: scores, split, np.zeros(6))
        # 0 et 2 sont exclus : le meilleur restant est la cible 1.
        self.assertAlmostEqual(r["ndcg@10"], 1.0)

    def test_utilisateurs_sans_cible_ignores(self):
        split = Split("val", input=matrice([[0], [0]]), target=matrice([[1], []]),
                      exclude=matrice([[0], [0]]))
        r = evaluer(lambda X: np.tile(np.arange(6.0)[::-1], (X.shape[0], 1)), split,
                    np.zeros(6))
        self.assertEqual(r["utilisateurs"], 1)


class FranchiseTests(unittest.TestCase):
    def test_titres_d_une_meme_franchise_partagent_leurs_mots(self):
        self.assertEqual(jetons_titre("Shingeki no Kyojin Season 2", "Attack on Titan Season 2"),
                         {"shingeki", "kyojin", "attack", "titan"})
        self.assertEqual(jetons_titre(None, "Death Note"), {"death", "note"})


if __name__ == "__main__":
    unittest.main()

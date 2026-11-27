import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from series import CatalogueSeries
from mal_profile import rated_series_profile


class SeriesTests(unittest.TestCase):
    def test_saisons_et_recaps_du_catalogue_item_knn(self):
        # Une note élevée sur une suite ou un récap doit classer sa série une seule fois.
        ids = np.array([16498, 59571, 40748, 56243, 40221, 59989])
        catalogue = CatalogueSeries(ids, SRC.parent / "data/content/series.csv")
        self.assertEqual(catalogue.series, [16498, 16498, 40748, 40748, 40221, 40221])
        scores = np.zeros((len(ids), len(ids)))
        scores[1] = [1, 1, .4, .9, .5, .8]
        self.assertEqual(catalogue.recommander(scores, [1], [59571]), [(2, 0), (4, 0)])
        self.assertEqual(catalogue.recommander(scores, [1], [59571, 56243]), [(4, 0)])

    def test_notes_mal_et_explication_knn(self):
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "series.csv"
            fichier.write_text("mal_id,series_id,entry_order\n1,1,0\n2,1,1\n")
            catalogue = CatalogueSeries(np.arange(1, 8), fichier)
        entries = {1: ("completed", 5), 2: ("completed", 0), 3: ("completed", 8),
                   4: ("completed", 10), 5: ("completed", 0), 6: ("dropped", 9)}
        indices, vues, preferences, _ = rated_series_profile(entries, catalogue)
        self.assertEqual(preferences, {3: 0.6, 4: 1.0})
        self.assertEqual(vues, {1, 3, 4, 5, 6})
        poids = np.zeros((7, 7))
        poids[:4, 6] = [1, 1, 0.9, 0.7]
        self.assertEqual(catalogue.recommander(poids, indices, vues, preferences=preferences), [(6, 3)])
        self.assertEqual(rated_series_profile({2: ("completed", 0)}, catalogue)[2], {1: 1.0})
        self.assertEqual(rated_series_profile({2: ("completed", 3)}, catalogue)[2], {})

    def test_regroupement_exclusions_et_contributions(self):
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "series.csv"
            fichier.write_text("mal_id,series_id,entry_order\n1,1,0\n2,1,1\n3,3,0\n4,3,1\n"
                               "5,5,0\n55,5,1\n", encoding="utf-8")
            catalogue = CatalogueSeries(np.arange(1, 8), fichier)
        poids = np.zeros((7, 7))
        # La saison 2 du candidat obtient le meilleur score de sa série.
        poids[0] = poids[1] = [100, 100, 1, 6, 100, 0, 9]
        poids[5, 3] = 10
        poids[5, 6] = 9
        # 55 est une saison déjà vue, absente du modèle. Elle exclut également 5.
        resultat = catalogue.recommander(poids, [0, 1, 5], [55])
        self.assertEqual(resultat, [(6, 0), (2, 5)])
        self.assertEqual(resultat, catalogue.recommander(poids, [0, 5], [55]))
        self.assertEqual(catalogue.recommander(poids, [1], np.arange(1, 8)), [])

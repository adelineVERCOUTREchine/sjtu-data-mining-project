import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import config


class DemoTests(unittest.TestCase):
    def test_modifier_le_profil_reclasse_et_explique_les_suggestions(self):
        titres = np.array([f"Anime {i}" for i in range(14)])
        poids = np.zeros((14, 14), dtype=np.float32)
        poids[0] = np.arange(14)
        poids[1, 3] = 20
        with tempfile.TemporaryDirectory() as dossier, patch.object(config, "MODELS_DIR", Path(dossier)):
            np.savez(Path(dossier) / "ease_demo.npz", weights=poids, mal_ids=1_000_000 + np.arange(14),
                     titles=titres, source="2025")
            app = AppTest.from_file(str(SRC / "demo.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)
            app.multiselect[0].select(0).run()
            premier = app.dataframe[0].value
            self.assertEqual(premier["Recommended anime"].tolist(), titres[13:3:-1].tolist())
            app.multiselect[0].select(1).run()
            second = app.dataframe[0].value
            self.assertEqual(len(second), 10)
            self.assertEqual(second.iloc[0]["Recommended anime"], "Anime 3")
            self.assertEqual(second.iloc[0]["Strongest contribution"], "Anime 1")
            self.assertTrue((second.iloc[1:]["Strongest contribution"] == "Anime 0").all())
            app.multiselect[0].set_value([0, 13]).run()
            self.assertNotIn("Anime 13", app.dataframe[0].value["Recommended anime"].tolist())
            app.multiselect[0].set_value([]).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minhash_genres_tags import (
    calculer_signatures,
    charger_documents,
    creer_fonctions_hachage,
    creer_labels,
    exporter_similarites,
)


class MinHashGenresTagsTests(unittest.TestCase):
    def test_labels_complets_normalises_et_sans_ordre(self):
        labels = creer_labels("Action| Slice of Life |ACTION", "School|Action|")
        self.assertEqual(labels, {"genre:action", "genre:slice of life", "tag:school", "tag:action"})
        self.assertEqual(labels, creer_labels("slice of life|action", "action|school"))

    def test_signatures_reproductibles_et_nombre_configurable(self):
        docs = {"1": creer_labels("Action|Drama", "School"),
                "2": creer_labels("Drama|Action", "School"),
                "3": creer_labels("Comedy", "")}
        fonctions = creer_fonctions_hachage(256)
        self.assertEqual(fonctions, creer_fonctions_hachage(256))
        signatures = calculer_signatures(docs, fonctions)
        self.assertTrue(all(len(s) == 256 for s in signatures))
        self.assertEqual(signatures[0], signatures[1])
        self.assertNotEqual(signatures[0], signatures[2])
        with self.assertRaises(ValueError):
            creer_fonctions_hachage(0)
        with self.assertRaises(ValueError):
            calculer_signatures({"vide": frozenset()}, fonctions)

    def test_estimation_sur_toutes_les_paires_et_jaccard_exact(self):
        # Ensembles identiques, disjoints et partiellement communs.
        docs = {"1": creer_labels("Action|Drama", ""),
                "2": creer_labels("Drama|Action", ""),
                "3": creer_labels("Comedy", ""),
                "4": creer_labels("Action", "School")}
        signatures = calculer_signatures(docs, creer_fonctions_hachage())
        estime = sum(a == b for a, b in zip(signatures[0], signatures[3])) / 128
        self.assertAlmostEqual(estime, 1 / 3, delta=0.12)
        with tempfile.TemporaryDirectory() as repertoire:
            sortie = Path(repertoire) / "prediction.csv"
            self.assertEqual(exporter_similarites(docs, signatures, sortie, 0.5), 1)
            with sortie.open(newline="", encoding="utf-8") as fichier:
                lignes = list(csv.DictReader(fichier))
            self.assertEqual(lignes, [{"article1": "1", "article2": "2", "Est. J": "1.0", "Act. J": "1.0"}])
            exporter_similarites(docs, signatures, sortie, 0)
            with sortie.open(newline="", encoding="utf-8") as fichier:
                lignes = list(csv.DictReader(fichier))
            partielle = next(l for l in lignes if l["article1"] == "1" and l["article2"] == "4")
            self.assertAlmostEqual(float(partielle["Act. J"]), 1 / 3)

    def test_lecture_champs_vides_et_label_unique(self):
        with tempfile.TemporaryDirectory() as repertoire:
            source = Path(repertoire) / "test.train"
            source.write_text("1\tSlice of Life\t\n2\t\tSchool\n3\t\t\n", encoding="utf-8")
            documents, ignores = charger_documents(source)
            self.assertEqual(ignores, 1)
            self.assertEqual(documents, {"1": {"genre:slice of life"}, "2": {"tag:school"}})


if __name__ == "__main__":
    unittest.main()

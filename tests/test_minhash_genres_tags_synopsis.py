import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minhash_common import creer_fonctions_hachage
from minhash_genres_tags import calculer_signatures as signatures_reference
from minhash_genres_tags_synopsis import (
    calculer_signatures,
    charger_documents,
    creer_shingles,
    main,
)


class MinHashCombineTests(unittest.TestCase):
    def test_labels_entiers_et_trigrammes_sans_melanger_les_champs(self):
        shingles = creer_shingles("Action|Slice of Life|ACTION", "School", "one two three four")
        self.assertEqual(shingles, {
            "genre:action", "genre:slice of life", "tag:school",
            "synopsis:one two three", "synopsis:two three four",
        })
        for genres, tags, texte in (("Comedy", "School", "one two three four"),
                                    ("Action|Slice of Life", "Military", "one two three four"),
                                    ("Action|Slice of Life", "School", "five six seven eight")):
            self.assertNotEqual(shingles, creer_shingles(genres, tags, texte))

    def test_memes_signatures_que_le_traitement_existant(self):
        documents = {"1": creer_shingles("Action", "School", "one two three four"),
                     "2": creer_shingles("Comedy", "School", "one two three four")}
        for nombre in (128, 256):
            fonctions = creer_fonctions_hachage(nombre)
            signatures = calculer_signatures(documents, fonctions)
            self.assertTrue(all(len(signature) == nombre for signature in signatures))
            self.assertEqual(signatures, signatures_reference(documents, fonctions))
        with self.assertRaises(ValueError):
            calculer_signatures({"vide": frozenset()}, creer_fonctions_hachage())

    def test_lecture_exclut_synopsis_courts_et_refuse_doublons(self):
        with tempfile.TemporaryDirectory() as repertoire:
            source = Path(repertoire) / "test.train"
            source.write_text("1\tAction\t\tone two three\n2\tDrama\tSchool\tone two\n", encoding="utf-8")
            documents, ignores = charger_documents(source)
            self.assertEqual(ignores, 1)
            self.assertEqual(documents, {"1": {"genre:action", "synopsis:one two three"}})
            source.write_text("1\tAction\t\tone two three\n1\tDrama\t\tone two three\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                charger_documents(source)

    def test_execution_et_jaccard_exact_des_trois_champs(self):
        with tempfile.TemporaryDirectory() as repertoire:
            source = Path(repertoire) / "test.train"
            sortie = Path(repertoire) / "resultat.csv"
            source.write_text(
                "1\tAction\tSchool\tone two three four\n"
                "2\tComedy\tSchool\tone two three four\n"
                "3\tDrama\tMilitary\tfive six seven eight\n", encoding="utf-8",
            )
            with patch.object(sys, "argv", ["minhash_genres_tags_synopsis.py", "--input", str(source),
                                           "--output", str(sortie), "--threshold", "0"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            with sortie.open(encoding="utf-8", newline="") as fichier:
                lignes = list(csv.DictReader(fichier))
            self.assertEqual(len(lignes), 1)
            self.assertEqual((lignes[0]["article1"], lignes[0]["article2"]), ("1", "2"))
            self.assertAlmostEqual(float(lignes[0]["Act. J"]), 3 / 5)
            self.assertAlmostEqual(float(lignes[0]["Est. J"]), 3 / 5, delta=0.15)


if __name__ == "__main__":
    unittest.main()

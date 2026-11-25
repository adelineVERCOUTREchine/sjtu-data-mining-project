import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.minhash import NUM_HASHES, creer_fonctions_hachage
from main import main
from methods.minhash_genres_tags import calculer_signatures as signatures_reference
from methods.minhash_synopsis import charger_documents, signatures_par_document


class MinHashSynopsisTests(unittest.TestCase):
    def test_shingles_de_trois_mots_et_synopsis_courts_ignores(self):
        with tempfile.TemporaryDirectory() as repertoire:
            source = Path(repertoire) / "test.train"
            source.write_text("1 one two three four\n2 one two\n", encoding="utf-8")
            documents, ignores = charger_documents(source)
            self.assertEqual(ignores, 1)
            self.assertEqual(documents, {"1": {"one two three", "two three four"}})

    def test_synopsis_et_metadonnees_utilisent_les_memes_128_fonctions(self):
        self.assertEqual(NUM_HASHES, 128)
        shingles = {"1": frozenset({"one two three", "two three four"}),
                    "3": frozenset({"five six seven", "six seven eight"})}
        fonctions = creer_fonctions_hachage()
        self.assertEqual(signatures_par_document(shingles, fonctions),
                         signatures_reference(shingles, fonctions))

    def test_execution_via_le_lanceur(self):
        with tempfile.TemporaryDirectory() as repertoire:
            source = Path(repertoire) / "anime_synopsis.train"
            sortie = Path(repertoire) / "prediction.csv"
            source.write_text(
                "1 one two three four\n2 one two three four\n3 five six seven eight\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                main(["minhash_synopsis", "--input", str(source), "--output", str(sortie)])
            with sortie.open(encoding="utf-8", newline="") as fichier:
                lignes = list(csv.DictReader(fichier))
            self.assertEqual(lignes, [{"article1": "1", "article2": "2",
                                       "Est. J": "1.0", "Act. J": "1.0"}])


if __name__ == "__main__":
    unittest.main()

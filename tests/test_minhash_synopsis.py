import contextlib
import csv
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from minhash_common import NUM_HASHES, creer_fonctions_hachage
from minhash_genres_tags import calculer_signatures


class MinHashSynopsisTests(unittest.TestCase):
    def test_synopsis_et_metadonnees_utilisent_les_memes_128_fonctions(self):
        with tempfile.TemporaryDirectory() as repertoire:
            projet = Path(repertoire)
            (projet / "archive").mkdir()
            (projet / "archive" / "anime_synopsis.train").write_text(
                "1 one two three four\n2 one two three four\n3 five six seven eight\n",
                encoding="utf-8",
            )
            # Exécuter le vrai traitement sur un petit corpus, sans toucher aux données.
            with patch.object(Path, "resolve", return_value=projet / "src" / "main.py"), \
                 contextlib.redirect_stdout(io.StringIO()):
                resultat = runpy.run_path(str(SRC / "main.py"), run_name="__main__")

            self.assertEqual(resultat["numHashes"], NUM_HASHES)
            self.assertEqual(NUM_HASHES, 128)
            self.assertEqual(list(zip(resultat["a_list"], resultat["b_list"])),
                             creer_fonctions_hachage())
            # Les mêmes shingles doivent donner les mêmes signatures dans les deux chemins.
            shingles = {"1": {"one two three", "two three four"},
                        "2": {"one two three", "two three four"},
                        "3": {"five six seven", "six seven eight"}}
            shingles = {identifiant: frozenset(valeurs) for identifiant, valeurs in shingles.items()}
            signatures = calculer_signatures(shingles, creer_fonctions_hachage())
            self.assertEqual([tuple(s) for s in resultat["signatures"]], signatures)
            with (projet / "data" / "prediction.csv").open(encoding="utf-8", newline="") as sortie:
                lignes = list(csv.DictReader(sortie))
            self.assertEqual(lignes, [{"article1": "1", "article2": "2",
                                      "Est. J": "1.0", "Act. J": "1.0"}])


if __name__ == "__main__":
    unittest.main()

import gzip
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mal_profile as mal


class MalProfileTests(unittest.TestCase):
    def test_export_et_statuts(self):
        rows = [(8, "Completed", 8), (2, "Dropped", 8), (3, "On-Hold", 0),
                (5, "Watching", 4), (7, "Plan to Watch", 0), (999, "Watching", 9)]
        xml = "<myanimelist>" + "".join(
            f"<anime><series_animedb_id>{i}</series_animedb_id><my_status>{s}</my_status>"
            f"<my_score>{v}</my_score></anime>" for i, s, v in rows) + "</myanimelist>"
        for data in (xml.encode(), gzip.compress(xml.encode())):
            entries = mal.read_export(data)
            self.assertEqual(mal.model_profile(entries, [2, 3, 5, 7, 8]), ([1, 4], [0, 1, 2, 4]))
        for data in (b"invalid", b"<myanimelist><manga/></myanimelist>", b"\x1f\x8b", b"<!DOCTYPE x><x/>"):
            with self.assertRaises(mal.ProfileError):
                mal.read_export(data)

    def test_callback_unique_et_isole(self):
        callbacks = mal.OAuthCallbacks()
        state, verifier, url = callbacks.start()
        other, _, _ = callbacks.start()
        params = parse_qs(urlsplit(url).query)
        self.assertEqual(params["code_challenge"], [verifier])
        self.assertEqual(params["state"], [state])
        self.assertFalse(callbacks.receive("unknown", "code", ""))
        self.assertTrue(callbacks.receive(state, "code", ""))
        self.assertFalse(callbacks.receive(state, "second", ""))
        self.assertIsNone(callbacks.take(other))
        self.assertEqual(callbacks.take(state), ("code", False))
        with self.assertRaises(mal.ProfileError):
            callbacks.take(state)
        callbacks.cancel(other)
        self.assertFalse(callbacks.receive(other, "late", ""))

    @patch.object(mal, "api_json")
    def test_compte_authentifie_et_pagination(self, api):
        def page(i, paging):
            return {"data": [{"node": {"id": i}, "list_status": {"status": "completed", "score": 8}}],
                    "paging": paging}
        api.side_effect = [{"access_token": "test-token"}, page(3, {"next": "next"}), page(9, {})]
        self.assertEqual(mal.import_account("code", "verifier"), {3: ("completed", 8), 9: ("completed", 8)})
        token, first, second = api.call_args_list
        self.assertEqual(token.kwargs["data"]["code_verifier"], "verifier")
        self.assertTrue(first.args[1].endswith("/@me/animelist"))
        self.assertEqual(first.kwargs["headers"], {"Authorization": "Bearer test-token"})
        self.assertEqual(second.kwargs["params"]["offset"], 1)

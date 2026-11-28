import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anime_chat as chat
import config


class ChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = chat.SearchCatalogue(config.ANIME_CSV)

    @patch.object(chat, "completion")
    def test_search_beyond_rankings_and_followup(self, call):
        call.side_effect = [{"keywords": "space bounty hunters", "max_episodes": 30},
                            {"answer": "Try this.", "picks": [{"id": 1, "why": "Space bounty hunters."}]}]
        history = [{"role": "user", "content": "I want science fiction."}]
        answer = chat.advise(self.catalogue, {1, 1535}, {"EASE": {1535: 1}}, {}, {}, {},
                             history, "Under 30 episodes, please.")
        context = call.call_args.args[1]
        self.assertEqual(context["previous_messages"], history)
        self.assertEqual(set(context["eligible_candidates"]), {1})
        self.assertIn("Cowboy Bebop", answer)

    @patch.object(chat, "completion")
    def test_refuse_unseen_candidate_id(self, call):
        call.side_effect = [{}, {"answer": "Try it.", "picks": [{"id": 1535, "why": "Example"}]}]
        answer = chat.advise(self.catalogue, {1}, {}, {}, {}, {}, [], "Recommend one.")
        self.assertIn("Cowboy Bebop", answer)
        self.assertNotIn("Death Note", answer)

    @patch.object(chat, "completion")
    def test_tentative_near_match_without_model_pick(self, call):
        call.side_effect = [{"keywords": "space", "max_episodes": 12, "limited_evidence": True},
                            {"answer": "Please clarify.", "picks": []}]
        answer = chat.advise(self.catalogue, set(), {}, {}, {}, {}, [], "Space, ideally 12 episodes.",
                             fallback_pool={1, 1535})
        for detail in ("Cowboy Bebop", "26", "at most 12", "tentative", "outside"):
            self.assertIn(detail, answer)
        self.assertNotIn("Please clarify", answer)

    @patch.object(chat, "completion")
    def test_greeting_does_not_force_a_pick(self, call):
        answer = chat.advise(self.catalogue, {1}, {}, {}, {}, {}, [], " Hi! ")
        self.assertNotIn("Cowboy Bebop", answer)
        call.assert_not_called()

    @patch("anime_chat.requests.post")
    def test_arcbench_request(self, post):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://api.arc-bench.com/v1",
                                       "MODEL": "deepseek-flash", "MODEL_REASONING": "low"}):
            post.return_value.status_code = 200
            post.return_value.json.return_value = {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}
            self.assertEqual(chat.completion("English", {}, "demo-key"), {"ok": True})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.arc-bench.com/v1/chat/completions")
        self.assertEqual(kwargs["json"]["reasoning_effort"], "low")
        self.assertEqual(kwargs["json"]["model"], "deepseek-flash")
        self.assertFalse(kwargs["allow_redirects"])

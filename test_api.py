import json
import os
import unittest

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from api import app
from recommendation_store import COLLECTION, ROOT


class TopPicksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = TestClient(app)
        cls.http.__enter__()
        cls.addClassCleanup(cls.http.__exit__, None, None, None)
        cls.store = app.state.store
        cls.contract = json.loads((ROOT / "contract.json").read_text())

    def test_index(self):
        self.store.index_titles()
        client = self.store.client
        self.assertEqual(client.count(COLLECTION, exact=True).count, 300)
        points, _ = client.scroll(COLLECTION, limit=300, with_vectors=True)
        self.assertEqual({p.payload["title_id"] for p in points}, {r["title_id"] for r in self.store.catalog})
        self.assertTrue(all(len(p.vector) == 128 for p in points))

    def test_known(self):
        member = os.getenv("SAMPLE_MEMBER", "M-10073")
        response = self.http.get(f"/recommendations/{member}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        Draft202012Validator(self.contract).validate(body)
        self.assertEqual(body["user_id"], member)
        picks = body["recommendations"]
        ids = {row["title_id"] for row in picks}
        self.assertEqual(len(ids), 10)
        self.assertIn("TT-0300", ids)
        scores = [row["score"] for row in picks]
        self.assertEqual(scores, sorted(scores, reverse=True))
        metadata = {row["title_id"]: row for row in self.store.catalog}
        played = {row["title_id"] for row in self.store.history[member]}
        self.assertTrue(ids.isdisjoint(played))
        for pick in picks:
            row = metadata[pick["title_id"]]
            self.assertEqual((pick["title_name"], pick["primary_genre"]), (row["title_name"], row["primary_genre"]))
        self.save_sample("known_member_200.json", body)

    def test_unknown(self):
        for member in ("M-99999", "malformed"):
            response = self.http.get(f"/recommendations/{member}")
            self.assertEqual(response.status_code, 404)
            body = response.json()
            Draft202012Validator(self.contract["$defs"]["ErrorResponse"]).validate(body)
            self.assertEqual(body["user_id"], member)
            if member == "M-99999":
                self.save_sample("unknown_member_404.json", body)

    def save_sample(self, name, body):
        if os.getenv("SAVE_RESPONSES") == "1":
            output = ROOT / "saved_responses"
            output.mkdir(exist_ok=True)
            (output / name).write_text(json.dumps(body, indent=2) + "\n")
            print("SAMPLE", name, json.dumps(body))

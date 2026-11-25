import inspect
import unittest
from pathlib import Path

import numpy as np

import train_two_tower
from clean_data import read_csv
from evaluate import mcnemar, score
from features import Encoder, examples, queries, title_features
from two_tower import init_params, loss_and_grads, top_k


def title(i, genre="Drama"):
    return {"title_id": i, "title_type": "film", "primary_genre": genre, "secondary_genre": "",
            "maturity_rating": "R", "runtime_minutes": "90", "release_date": "2020-01-01"}


def watch(t, day, member="M", country="US"):
    return {"member_id": member, "title_id": t, "event_datetime": f"2026-11-{day:02d}", "country": country,
            "profile_type": "standard", "device_type": "tv"}


CATALOG = [title("B", "Crime"), title("A"), title("C")]


class Features(unittest.TestCase):
    def test_titles_are_sorted_and_blank_secondary_genre_is_not_a_feature(self):
        ids, x = title_features(CATALOG)
        self.assertEqual((ids, x.shape), (["A", "B", "C"], (3, 2 + 1 + 1 + 2)))

    def test_examples_use_only_earlier_rows(self):
        rows = [watch("C", 3), watch("A", 1), watch("B", 2)]
        enc = Encoder(rows, CATALOG)
        data = examples(rows, enc)
        self.assertEqual(data["answer"].tolist(), [1, 2])
        self.assertEqual(data["seen"].tolist(), [[True, False, False], [True, True, False]])
        changed = examples([watch("C", 3, country="BR"), watch("A", 1), watch("B", 2)], Encoder(rows, CATALOG))
        self.assertTrue(np.array_equal(data["members"], changed["members"]))

    def test_rewatch_and_title_outside_the_catalog_are_not_answers(self):
        rows = [watch("A", 1), watch("A", 2), watch("X", 3), watch("B", 4)]
        self.assertEqual(examples(rows, Encoder(rows, CATALOG))["answer"].tolist(), [1])


class Model(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        x, titles = rng.normal(size=(5, 4)), rng.normal(size=(6, 3))
        seen = np.zeros((5, 6), dtype=bool)
        seen[:, 0] = True
        answer = np.array([1, 2, 3, 4, 5])
        p = init_params(4, 3, 6, 2, 3, np.zeros(6), 0)
        _, grads = loss_and_grads(p, x, titles, seen, answer)
        for k in p:
            for _ in range(4):
                at = tuple(rng.integers(n) for n in p[k].shape)
                old, step = p[k][at], 1e-6
                p[k][at] = old + step
                up = loss_and_grads(p, x, titles, seen, answer)[0]
                p[k][at] = old - step
                down = loss_and_grads(p, x, titles, seen, answer)[0]
                p[k][at] = old
                self.assertAlmostEqual((up - down) / (2 * step), grads[k][at], places=5, msg=k)

    def test_top_k_skips_seen_and_breaks_ties_by_lower_index(self):
        self.assertEqual(top_k(np.array([[9.0, 1, 1, 0]]), np.array([[True, False, False, False]]), 2).tolist(), [[1, 2]])


    def test_non_finite_model_stops(self):
        p = init_params(4, 3, 6, 2, 3, np.zeros(6), 0)
        p["bias"][0] = np.nan
        with self.assertRaises(FloatingPointError):
            loss_and_grads(p, np.ones((1, 4)), np.ones((6, 3)), np.zeros((1, 6), dtype=bool), np.array([1]))


class Recommendations(unittest.TestCase):
    """Decision 9: only catalog titles. Watched titles are skipped too."""

    def setUp(self):
        catalog = [title(f"T{i:02d}", "Drama" if i % 2 else "Crime") for i in range(20)]
        self.rows = [watch("T01", 1), watch("X", 2), watch("T03", 3), watch("T00", 4)]
        self.encoder, self.catalog = Encoder(self.rows, catalog), catalog
        self.watched = {"T00", "T01", "T03", "X"}

    def check(self, titles):
        self.assertEqual(len(set(titles)), 10)
        self.assertLessEqual(set(titles), set(self.encoder.ids) - self.watched)

    def test_model(self):
        members, seen = queries(self.rows, self.encoder, ["M"])
        p = train_two_tower.new_model(self.encoder, {"dim": 4, "hidden": 4}, 0)
        self.check(train_two_tower.recommend(p, self.encoder, members, seen, ["M"])["M"])

    def test_genre_rule_puts_the_favourite_genre_first(self):
        titles = train_two_tower.genre_lists(self.rows, self.encoder, self.catalog, ["M"])["M"]
        self.check(titles)
        self.assertEqual(titles[:8], ["T05", "T07", "T09", "T11", "T13", "T15", "T17", "T19"])

    def test_tune_never_touches_the_holdout(self):
        self.assertNotIn("holdout", inspect.getsource(train_two_tower.tune))


ROOT = Path(__file__).parent
CATALOG_FILE = ROOT / "data" / "Netflix title catalog, November 2026.csv"


@unittest.skipUnless(CATALOG_FILE.exists(), "the raw files are not in data/")
class SavedModel(unittest.TestCase):
    def test_titles_are_the_catalog(self):
        saved = np.load(ROOT / "models" / "two_tower.npz")["title_ids"].tolist()
        self.assertEqual(saved, sorted(r["title_id"] for r in read_csv(CATALOG_FILE)))


class Evaluate(unittest.TestCase):
    def test_hit_and_ndcg(self):
        result = score({"M": ["A", "B"], "N": ["A", "B"]}, {"M": "A", "N": "Z"})
        self.assertEqual((result["hits"], result["hit"], result["ndcg"]), ([True, False], 0.5, 0.5))

    def test_mcnemar(self):
        self.assertEqual(mcnemar([True] * 6, [False] * 6), (6, 0, 2 / 64))
        self.assertEqual(mcnemar([True], [True]), (0, 0, 1.0))


if __name__ == "__main__":
    unittest.main()

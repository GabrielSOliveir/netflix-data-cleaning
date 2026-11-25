import unittest
from pathlib import Path

from clean_data import read_csv
from popular_top10 import catalog_ids, check_no_holdout, count_views, top_titles


def count(*titles):
    return count_views([{"t": t} for t in titles], "t")


def events(*ids):
    return [{"e": i, "t": "SECRET"} for i in ids]


class Holdout(unittest.TestCase):
    def test_no_shared_rows_passes(self):
        check_no_holdout(events("1", "2"), events("3"), "e")

    def test_holdout_row_fails_without_naming_the_title(self):
        with self.assertRaises(ValueError) as error:
            check_no_holdout(events("1", "2"), events("2"), "e")
        self.assertEqual(str(error.exception), "1 holdout rows in the training data")

    def test_empty_holdout_fails(self):
        with self.assertRaises(ValueError):
            check_no_holdout(events("1"), [], "e")


class CountViews(unittest.TestCase):
    def test_each_row_is_one_view(self):
        self.assertEqual(count("A", "A", "B"), {"A": 2, "B": 1})

    def test_blank_title_fails(self):
        with self.assertRaises(ValueError):
            count(" ")


class TopTitles(unittest.TestCase):
    def test_order_ties_and_cut(self):
        top = top_titles({"C": 2, "B": 2, "A": 1, "D": 5}, 3, set("ABCD"))
        self.assertEqual([(r["rank"], r["title_id"], r["views"]) for r in top], [(1, "D", 5), (2, "B", 2), (3, "C", 2)])

    def test_too_few_titles_fails(self):
        with self.assertRaises(ValueError):
            top_titles({"A": 1}, 2, {"A"})

    def test_title_not_in_catalog_is_replaced_by_the_next(self):
        top = top_titles({"X": 9, "A": 3, "B": 2, "C": 1}, 2, {"A", "B", "C"})
        self.assertEqual([(r["rank"], r["title_id"]) for r in top], [(1, "A"), (2, "B")])

    def test_titles_outside_the_catalog_do_not_fill_the_list(self):
        with self.assertRaises(ValueError):
            top_titles({"A": 1, "X": 9}, 2, {"A"})


class Catalog(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(catalog_ids([{"t": "A"}, {"t": "B"}], "t"), {"A", "B"})

    def test_empty_catalog_fails(self):
        with self.assertRaises(ValueError):
            catalog_ids([], "t")


DATA = Path(__file__).parent / "data"
CATALOG = DATA / "Netflix title catalog, November 2026.csv"


class SavedList(unittest.TestCase):
    """Failures show counts only, so no holdout title is ever printed."""

    def setUp(self):
        self.train = read_csv(DATA / "clean" / "train.csv")
        self.top = read_csv(DATA / "baseline" / "popular_top10.csv")

    def test_train_has_no_holdout_rows(self):
        check_no_holdout(self.train, read_csv(DATA / "clean" / "holdout.csv"), "event_id")

    def test_views_match_the_current_train(self):
        views = count_views(self.train, "title_id")
        self.assertEqual(sum(int(r["views"]) != views[r["title_id"]] for r in self.top), 0)

    @unittest.skipUnless(CATALOG.exists(), "the raw files are not in data/")
    def test_list_is_the_top_10_of_the_current_train(self):
        expected = top_titles(count_views(self.train, "title_id"), 10, catalog_ids(read_csv(CATALOG), "title_id"))
        self.assertEqual([r["title_id"] for r in self.top], [r["title_id"] for r in expected])


if __name__ == "__main__":
    unittest.main()

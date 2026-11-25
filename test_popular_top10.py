import unittest
from datetime import date

from popular_top10 import catalog_ids, count_views, top_titles


def count(*rows):
    return count_views([{"t": t, "d": d} for t, d in rows], "t", "d", date(2026, 11, 9))


class CountViews(unittest.TestCase):
    def test_each_row_is_one_view(self):
        self.assertEqual(count(("A", "2026-11-09"), ("A", "2026-08-17"), ("B", "2026-11-01")), {"A": 2, "B": 1})

    def test_holdout_row_fails(self):
        with self.assertRaises(ValueError):
            count(("A", "2026-11-09"), ("A", "2026-11-10"))

    def test_blank_title_fails(self):
        with self.assertRaises(ValueError):
            count((" ", "2026-11-01"))


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


if __name__ == "__main__":
    unittest.main()

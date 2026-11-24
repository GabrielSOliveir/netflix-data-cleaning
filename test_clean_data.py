import unittest
from datetime import date

from clean_data import clean_event, runtime_by_title, split_by_date, to_number


def run(watch, completed="True", runtime=6000.0):
    row = {"title_id": "T", "watch_seconds": watch, "completed": completed}
    return clean_event(row, runtime, "watch_seconds", "completed")


class CleanEvent(unittest.TestCase):
    def test_above_runtime_is_removed(self):
        for watch in ("6001", "85372"):
            self.assertEqual(run(watch), (None, "removed: above runtime"))

    def test_blank_is_removed(self):
        for watch in ("", "  "):
            self.assertEqual(run(watch), (None, "removed: blank watch time"))

    def test_blank_without_runtime_is_removed(self):
        self.assertIsNone(run("", runtime=None)[0])

    def test_exact_runtime_is_complete_whatever_the_original_says(self):
        for original in ("True", "False"):
            row, _ = run("6000", completed=original)
            self.assertEqual((row["completed"], row["completed_raw"]), ("True", original))

    def test_below_runtime_is_not_complete(self):
        row, _ = run("5999")
        self.assertEqual((row["completed"], row["completed_raw"]), ("False", "True"))

    def test_unknown_runtime_is_kept_but_not_complete(self):
        row, _ = run("99999", runtime=None)
        self.assertEqual((row["completed"], row["completed_check"]), ("False", "no_runtime"))

    def test_watch_time_is_never_changed(self):
        self.assertEqual(run("5999")[0]["watch_seconds"], "5999")


class Inputs(unittest.TestCase):
    def test_bad_numbers_fail(self):
        for text in ("abc", "nan", "inf", "-1"):
            with self.assertRaises(ValueError):
                to_number(text, "x")

    def test_runtime_in_seconds_and_blank(self):
        rows = [{"id": "A", "m": "2"}, {"id": "B", "m": ""}]
        self.assertEqual(runtime_by_title(rows, "id", "m", 60), {"A": 120, "B": None})

    def test_duplicate_title_fails(self):
        with self.assertRaises(ValueError):
            runtime_by_title([{"id": "A", "m": "2"}] * 2, "id", "m", 60)


class Split(unittest.TestCase):
    def test_boundaries(self):
        days = ["2026-11-09", "2026-11-10", "2026-11-17"]
        train, holdout = split_by_date([{"d": d} for d in days], "d", date(2026, 11, 9))
        self.assertEqual([[r["d"] for r in p] for p in (train, holdout)], [days[:1], days[1:]])


if __name__ == "__main__":
    unittest.main()

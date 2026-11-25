import unittest
from pathlib import Path

from clean_data import clean_event, read_csv, runtime_by_title, split_by_member, to_number


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


def split(*watches):
    """Split (member, title, date) tuples; titles starting with X are not in the catalog."""
    rows = [{"m": m, "t": t, "d": d} for m, t, d in watches]
    catalog = {t for _, t, _ in watches if not t.startswith("X")}
    train, holdout, repeats, later, left_out = split_by_member(rows, catalog, "m", "t", "d")
    return [[(r["t"], r["d"]) for r in part] for part in (train, holdout, repeats + later)] + [dict(left_out)]


class Split(unittest.TestCase):
    def test_last_new_title_is_holdout(self):
        self.assertEqual(split(("M", "B", "2026-11-02"), ("M", "A", "2026-11-01")),
                         [[("A", "2026-11-01")], [("B", "2026-11-02")], [], {}])

    def test_title_watched_before_is_not_holdout(self):
        train, holdout, removed, _ = split(("M", "A", "2026-11-01"), ("M", "B", "2026-11-02"), ("M", "A", "2026-11-03"))
        self.assertEqual((train, holdout), ([("A", "2026-11-01")], [("B", "2026-11-02")]))
        self.assertEqual(removed, [("A", "2026-11-03")])

    def test_title_outside_the_catalog_is_not_holdout(self):
        train, holdout, removed, _ = split(("M", "A", "2026-11-01"), ("M", "B", "2026-11-02"), ("M", "X", "2026-11-03"))
        self.assertEqual((train, holdout), ([("A", "2026-11-01")], [("B", "2026-11-02")]))
        self.assertEqual(removed, [("X", "2026-11-03")])

    def test_same_day_row_that_ranks_later_is_removed(self):
        train, holdout, removed, _ = split(("M", "A", "2026-11-01"), ("M", "B", "2026-11-02"), ("M", "X", "2026-11-02"))
        self.assertEqual((train, holdout, removed), ([("A", "2026-11-01")], [("B", "2026-11-02")], [("X", "2026-11-02")]))

    def test_later_repeat_of_the_holdout_title_is_removed(self):
        train, holdout, removed, _ = split(("M", "A", "2026-11-01"), ("M", "B", "2026-11-02"), ("M", "B", "2026-11-03"))
        self.assertEqual((train, holdout, removed), ([("A", "2026-11-01")], [("B", "2026-11-02")], [("B", "2026-11-03")]))

    def test_same_timestamp_goes_to_the_higher_title_id(self):
        for order in (("A", "B"), ("B", "A")):
            holdout = split(*[("M", t, "2026-11-01") for t in order])[1]
            self.assertEqual(holdout, [("B", "2026-11-01")])

    def test_time_of_day_counts_before_title_id(self):
        holdout = split(("M", "B", "2026-11-01T08:00:00"), ("M", "A", "2026-11-01T09:00:00"))[1]
        self.assertEqual(holdout, [("A", "2026-11-01T09:00:00")])

    def test_members_left_out_stay_whole_in_train(self):
        one = ("M", "A", "2026-11-01")
        for watches, reason in (([one], "no watch before the holdout title"),
                                ([one, ("M", "A", "2026-11-02")], "no watch before the holdout title"),
                                ([one, ("M", "X", "2026-11-02")], "no watch before the holdout title"),
                                ([("M", "X1", "2026-11-01"), ("M", "X2", "2026-11-02")], "no catalog title")):
            train, holdout, removed, left_out = split(*watches)
            self.assertEqual((len(train), holdout, removed, left_out), (len(watches), [], [], {reason: 1}))

    def test_members_do_not_affect_each_other(self):
        holdout = split(("M", "A", "2026-11-01"), ("M", "B", "2026-11-02"),
                        ("N", "B", "2026-11-01"), ("N", "A", "2026-11-03"))[1]
        self.assertEqual(holdout, [("B", "2026-11-02"), ("A", "2026-11-03")])

    def test_blank_member_fails(self):
        with self.assertRaises(ValueError):
            split((" ", "A", "2026-11-01"))


DATA = Path(__file__).parent / "data"
CATALOG = DATA / "Netflix title catalog, November 2026.csv"
EVENTS = DATA / "Netflix viewing events export, 17 Aug to 16 Nov 2026.csv"


class SavedFiles(unittest.TestCase):
    """Failures show counts only, so no holdout title is ever printed."""

    def setUp(self):
        self.train = read_csv(DATA / "clean" / "train.csv")
        self.holdout = read_csv(DATA / "clean" / "holdout.csv")

    def test_one_holdout_row_per_member(self):
        self.assertEqual(len({r["member_id"] for r in self.holdout}), len(self.holdout))

    def test_every_holdout_member_has_a_train_row(self):
        self.assertEqual(len({r["member_id"] for r in self.holdout} - {r["member_id"] for r in self.train}), 0)

    def test_no_holdout_title_in_the_members_own_train(self):
        pairs = {(r["member_id"], r["title_id"]) for r in self.train}
        self.assertEqual(sum((r["member_id"], r["title_id"]) in pairs for r in self.holdout), 0)

    def test_no_train_row_is_later_than_its_members_holdout_watch(self):
        last = {r["member_id"]: (r["event_datetime"], r["title_id"]) for r in self.holdout}
        self.assertEqual(sum((r["event_datetime"], r["title_id"]) >= last[r["member_id"]]
                             for r in self.train if r["member_id"] in last), 0)

    def test_no_event_in_both_files(self):
        ids = [r["event_id"] for r in self.train + self.holdout]
        self.assertEqual(len(set(ids)), len(ids))


@unittest.skipUnless(CATALOG.exists() and EVENTS.exists(), "the raw files are not in data/")
class SavedFilesAgainstRaw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_csv(DATA / "clean" / "train.csv")
        cls.holdout = read_csv(DATA / "clean" / "holdout.csv")
        cls.runtimes = runtime_by_title(read_csv(CATALOG), "title_id", "runtime_minutes", 60)
        rows = [clean_event(r, cls.runtimes.get(r["title_id"]), "watch_seconds", "completed")[0] for r in read_csv(EVENTS)]
        cls.clean = [r for r in rows if r]

    def expected(self):
        """Latest eligible event id per member, found without split_by_member."""
        result = {}
        for member in {r["member_id"] for r in self.clean}:
            own = sorted((r for r in self.clean if r["member_id"] == member),
                         key=lambda r: (r["event_datetime"], r["title_id"]))
            for i in reversed(range(len(own))):
                title = own[i]["title_id"]
                if title in self.runtimes and title not in [r["title_id"] for r in own[:i]]:
                    if i > 0:
                        result[member] = own[i]["event_id"]
                    break
        return result

    def test_every_holdout_row_is_the_latest_eligible_watch(self):
        saved = {r["member_id"]: r["event_id"] for r in self.holdout}
        expected = self.expected()
        self.assertEqual((len(saved), sum(saved[m] != expected.get(m) for m in saved)), (len(expected), 0))

    def test_train_holdout_and_removed_add_up_to_the_cleaned_total(self):
        last = {r["member_id"]: (r["event_datetime"], r["title_id"]) for r in self.holdout}
        kept = {r["event_id"] for r in self.train + self.holdout}
        missing = [r for r in self.clean if r["event_id"] not in kept]
        _, _, repeats, later, _ = split_by_member(self.clean, set(self.runtimes), "member_id", "title_id", "event_datetime")
        self.assertEqual(len(self.train) + len(self.holdout) + len(repeats) + len(later), len(self.clean))
        self.assertEqual(len(missing), len(repeats) + len(later))
        self.assertEqual(sum(r["member_id"] not in last for r in missing), 0)
        self.assertEqual(sum((r["event_datetime"], r["title_id"]) < last[r["member_id"]] for r in missing), 0)

    def test_saved_rows_equal_the_cleaned_rows(self):
        clean = {r["event_id"]: r for r in self.clean}
        self.assertEqual(sum(clean.get(r["event_id"]) != r for r in self.train + self.holdout), 0)


if __name__ == "__main__":
    unittest.main()

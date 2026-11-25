"""Clean viewing events with a title catalog, then split them per member.

Rules, applied to each event:
  1. Watch time above the title's runtime: remove the record.
  2. Blank watch time: remove the record.
  3. Completed is True only when the watch time equals the catalog runtime.
     With no known runtime the match cannot be proved, so completed is False.

Split, for each member, with watches sorted by date and time, then title id, then file order:
  holdout is the last watch of a catalog title the member had not watched before;
  train is every watch before it. Every watch after it is removed.
  A member without one holdout title and one watch before it stays whole in train.
"""
import argparse
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

AUDIT_COLUMNS = ["completed_raw", "completed_check"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def to_number(text, where):
    """Return a float, or None for a blank. Anything else is an error."""
    if not text.strip():
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{where}: {text!r} is not a number")
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{where}: {text!r} is not a valid duration")
    return number


def runtime_by_title(catalog, id_col, runtime_col, seconds_per_unit):
    """Map each title id to its runtime in seconds (None when unknown)."""
    runtimes = {}
    for row in catalog:
        title = row[id_col]
        if title in runtimes:
            raise ValueError(f"catalog: {title} appears more than once")
        runtime = to_number(row[runtime_col], f"catalog {title}")
        runtimes[title] = runtime * seconds_per_unit if runtime else None
    return runtimes


def clean_event(row, runtime, watch_col, completed_col):
    """Return (clean row or None, outcome). None means: remove the record."""
    watch = to_number(row[watch_col], f"event {row}")
    if watch is None:
        return None, "removed: blank watch time"
    if runtime is not None and watch > runtime:
        return None, "removed: above runtime"
    check = "no_runtime" if runtime is None else "verified"
    completed = "True" if watch == runtime else "False"
    return dict(row, **{"completed_raw": row[completed_col], "completed_check": check,
                        completed_col: completed}), f"kept: {check}"


def split_by_member(rows, catalog, member_col, id_col, date_col):
    """Return (train, holdout, removed repeats, removed later watches, left out members by reason)."""
    by_member = {}
    for i, row in enumerate(rows):
        if not row[member_col].strip():
            raise ValueError(f"row with no member: {i}")
        by_member.setdefault(row[member_col], []).append(i)

    holdout, repeats, later, left_out = set(), set(), set(), Counter()
    for watches in by_member.values():
        watches.sort(key=lambda i: (datetime.fromisoformat(rows[i][date_col]), rows[i][id_col]))
        seen, pick = set(), None
        for i in watches:
            title = rows[i][id_col]
            if title in catalog and title not in seen:
                pick = i
            seen.add(title)
        if pick is None:
            left_out["no catalog title"] += 1
            continue
        if pick == watches[0]:
            left_out["no watch before the holdout title"] += 1
            continue
        holdout.add(pick)
        for i in watches[watches.index(pick) + 1:]:
            (repeats if rows[i][id_col] == rows[pick][id_col] else later).add(i)

    train = set(range(len(rows))) - holdout - repeats - later
    return tuple([rows[i] for i in sorted(part)] for part in (train, holdout, repeats, later)) + (left_out,)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--out", required=True, help="folder for the clean files")
    p.add_argument("--member-col", default="member_id")
    p.add_argument("--id-col", default="title_id")
    p.add_argument("--runtime-col", default="runtime_minutes")
    p.add_argument("--seconds-per-unit", type=float, default=60, help="60 when the runtime is in minutes")
    p.add_argument("--watch-col", default="watch_seconds")
    p.add_argument("--completed-col", default="completed")
    p.add_argument("--date-col", default="event_datetime")
    return p.parse_args()


def main():
    a = parse_args()
    events = read_csv(a.events)
    if not events:
        raise SystemExit("The events file has no rows")
    runtimes = runtime_by_title(read_csv(a.catalog), a.id_col, a.runtime_col, a.seconds_per_unit)

    clean, outcomes = [], Counter()
    for row in events:
        new, outcome = clean_event(row, runtimes.get(row[a.id_col]), a.watch_col, a.completed_col)
        outcomes[outcome] += 1
        if new:
            clean.append(new)
    train, holdout, repeats, later, left_out = split_by_member(clean, set(runtimes), a.member_col, a.id_col, a.date_col)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("holdout", holdout)):
        write_csv(out / f"{name}.csv", rows, list(events[0]) + AUDIT_COLUMNS)

    print(f"read {len(events)} | clean {len(clean)} | members {len({r[a.member_col] for r in clean})}")
    print(f"train {len(train)} | holdout {len(holdout)} | removed after the holdout {len(repeats) + len(later)} "
          f"(repeats of the holdout title {len(repeats)}, other titles {len(later)})")
    print(f"members left out of holdout: {sum(left_out.values())} {dict(left_out)}")
    for outcome, n in sorted(outcomes.items()):
        print(f"  {outcome}: {n}")


if __name__ == "__main__":
    main()

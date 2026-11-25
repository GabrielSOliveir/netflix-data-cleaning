"""Save the most watched titles in the training data: one list, the same for every member.

A view is one row. Ties are broken by title id. Only titles in the catalog can enter the list.
Any row that is also in the holdout stops the script, so holdout data cannot enter the counts.
"""
import argparse
from collections import Counter
from pathlib import Path

from clean_data import read_csv, write_csv


def check_no_holdout(rows, holdout, event_col):
    """Compare event ids only. The message has a count, never a holdout title."""
    if not holdout:
        raise ValueError("the holdout has no rows to check against")
    held = {row[event_col] for row in holdout}
    used = sum(row[event_col] in held for row in rows)
    if used:
        raise ValueError(f"{used} holdout rows in the training data")


def count_views(rows, id_col):
    views = Counter()
    for row in rows:
        if not row[id_col].strip():
            raise ValueError(f"row with no title: {row}")
        views[row[id_col]] += 1
    return views


def catalog_ids(catalog, id_col):
    ids = {row[id_col] for row in catalog}
    if not ids:
        raise ValueError("the catalog has no titles")
    return ids


def top_titles(views, n, allowed):
    views = {title: count for title, count in views.items() if title in allowed}
    if len(views) < n:
        raise ValueError(f"only {len(views)} catalog titles, {n} needed")
    ranked = sorted(views.items(), key=lambda item: (-item[1], item[0]))[:n]
    return [{"rank": i, "title_id": t, "views": v} for i, (t, v) in enumerate(ranked, 1)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--holdout", required=True, help="only read to check that none of its rows are used")
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True, help="csv file to write")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--id-col", default="title_id")
    p.add_argument("--event-col", default="event_id")
    return p.parse_args()


def main():
    a = parse_args()
    rows = read_csv(a.train)
    check_no_holdout(rows, read_csv(a.holdout), a.event_col)
    views = count_views(rows, a.id_col)
    allowed = catalog_ids(read_csv(a.catalog), a.id_col)
    top = top_titles(views, a.n, allowed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, top, ["rank", "title_id", "views"])
    print(f"read {len(rows)} | titles not in catalog {len(set(views) - allowed)} | saved {len(top)} titles to {out}")


if __name__ == "__main__":
    main()

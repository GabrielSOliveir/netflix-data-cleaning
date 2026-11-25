"""Turn train rows and the catalog into arrays. No holdout row ever reaches this module."""
from collections import Counter, defaultdict
from datetime import datetime
from math import log, log1p

import numpy as np

CONTEXT = ["country", "profile_type", "device_type"]
MIN_RUNTIME = 1.0


def title_features(catalog):
    """Return (sorted title ids, one row of features per title)."""
    catalog = sorted(catalog, key=lambda r: r["title_id"])
    genres = sorted({r["primary_genre"] for r in catalog} | {r["secondary_genre"] for r in catalog} - {""})
    types = sorted({r["title_type"] for r in catalog})
    ratings = sorted({r["maturity_rating"] for r in catalog})
    out = []
    for r in catalog:
        runtime = float(r["runtime_minutes"] or 0)
        out.append([1.0 if r["primary_genre"] == g else 0.5 if r["secondary_genre"] == g else 0.0 for g in genres]
                   + [float(r["title_type"] == t) for t in types]
                   + [float(r["maturity_rating"] == t) for t in ratings]
                   + [log(max(runtime, MIN_RUNTIME)) / 5 if runtime else 0.0,
                      (int(r["release_date"][:4]) - 2015) / 10 if r["release_date"] else 0.0])
    return [r["title_id"] for r in catalog], np.array(out)


class Encoder:
    """Member features from a list of that member's past rows."""

    def __init__(self, rows, catalog):
        self.ids, self.titles = title_features(catalog)
        self.index = {t: i for i, t in enumerate(self.ids)}
        self.context = [(c, v) for c in CONTEXT for v in sorted({r[c] for r in rows})]
        self.slot = {cv: i for i, cv in enumerate(self.context)}
        views = Counter(r["title_id"] for r in rows)
        self.log_pop = np.array([log(views[t] + 1.0) for t in self.ids])
        self.n_member = self.titles.shape[1] + len(self.context) + 1

    def member(self, history):
        known = [self.index[r["title_id"]] for r in history if r["title_id"] in self.index]
        taste = self.titles[known].mean(0) if known else np.zeros(self.titles.shape[1])
        context = np.zeros(len(self.context))
        for r in history:
            for c in CONTEXT:
                if (c, r[c]) in self.slot:
                    context[self.slot[(c, r[c])]] += 1 / len(history)
        return np.concatenate([taste, context, [log1p(len(history)) / 3]])

    def seen(self, history):
        mask = np.zeros(len(self.ids), dtype=bool)
        mask[[self.index[r["title_id"]] for r in history if r["title_id"] in self.index]] = True
        return mask


def histories(rows):
    by_member = defaultdict(list)
    for r in rows:
        by_member[r["member_id"]].append(r)
    for own in by_member.values():
        own.sort(key=lambda r: (datetime.fromisoformat(r["event_datetime"]), r["title_id"]))
    return by_member


def examples(rows, encoder):
    """One example per first watch of a catalog title, using only the rows before it."""
    members, seen, answer = [], [], []
    for own in histories(rows).values():
        for i in range(1, len(own)):
            title = own[i]["title_id"]
            if title in encoder.index and title not in {r["title_id"] for r in own[:i]}:
                members.append(encoder.member(own[:i]))
                seen.append(encoder.seen(own[:i]))
                answer.append(encoder.index[title])
    return {"members": np.array(members), "seen": np.array(seen), "answer": np.array(answer)}


def queries(rows, encoder, member_ids):
    by_member = histories(rows)
    return (np.array([encoder.member(by_member[m]) for m in member_ids]),
            np.array([encoder.seen(by_member[m]) for m in member_ids]))

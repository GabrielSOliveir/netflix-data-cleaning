"""Train the two-tower model.

tune:  choose the settings on a validation set cut from train. It has no access to the holdout.
final: train on all of train with the chosen settings and read the holdout once.
"""
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from clean_data import read_csv, split_by_member
from evaluate import mcnemar, score
from features import Encoder, examples, histories, queries
from popular_top10 import check_no_holdout, count_views, top_titles
from two_tower import forward, init_params, top_k, train

K = 10
SEEDS = range(5)
GRID = {"dim": [8, 16], "hidden": [16, 32], "lr": [0.003, 0.01], "decay": [0.1, 0.5, 1.0, 2.0]}
MAX_EPOCHS, EVERY = 60, 5


def lists_from(encoder, top, member_ids):
    return {m: [encoder.ids[i] for i in row] for m, row in zip(member_ids, top)}


def baselines(rows, encoder, catalog, member_ids, seen, saved=None):
    """Return the popular list as it is, without watched titles, and by favourite genre."""
    views = count_views(rows, "title_id")
    fixed = saved or [r["title_id"] for r in top_titles(views, K, set(encoder.ids))]
    popular = np.tile(np.array([float(views[t]) for t in encoder.ids]), (len(member_ids), 1))
    return {"popular": {m: fixed for m in member_ids},
            "popular unwatched": lists_from(encoder, top_k(popular, seen, K), member_ids),
            "popular in favourite genre": genre_lists(rows, encoder, catalog, member_ids)}


def genre_lists(rows, encoder, catalog, member_ids):
    """No learning: unwatched titles by views of the genre by the member, then by views of the title."""
    genre = {r["title_id"]: r["primary_genre"] for r in catalog}
    views = count_views(rows, "title_id")
    by_member = histories(rows)
    lists = {}
    for m in member_ids:
        watched = [r["title_id"] for r in by_member[m] if r["title_id"] in genre]
        taste = Counter(genre[t] for t in watched)
        unseen = [t for t in encoder.ids if t not in set(watched)]
        lists[m] = sorted(unseen, key=lambda t: (-taste[genre[t]], -views[t], t))[:K]
    return lists


def new_model(encoder, s, seed):
    return init_params(encoder.n_member, encoder.titles.shape[1], len(encoder.ids), s["dim"], s["hidden"],
                       encoder.log_pop, seed)


def recommend(p, encoder, members, seen, member_ids):
    return lists_from(encoder, top_k(forward(p, members, encoder.titles)[0], seen, K), member_ids)


def tune(a):
    rows, catalog = read_csv(a.train), read_csv(a.catalog)
    fit, val, _, _, _ = split_by_member(rows, {r["title_id"] for r in catalog}, "member_id", "title_id", "event_datetime")
    encoder = Encoder(fit, catalog)
    data = examples(fit, encoder)
    answers = {r["member_id"]: r["title_id"] for r in val}
    ids = list(answers)
    members, seen = queries(fit, encoder, ids)
    print(f"fit {len(fit)} rows | {len(data['answer'])} examples | validation {len(ids)} members")
    for name, lists in baselines(fit, encoder, catalog, ids, seen).items():
        r = score(lists, answers)
        print(f"  {name}: hit@{K} {r['hit']:.4f} ndcg {r['ndcg']:.4f}")

    results = []
    for values in product(*GRID.values()):
        s = dict(zip(GRID, values))
        curve = {}
        for seed in SEEDS:
            def check(epoch, p):
                if epoch % EVERY == 0:
                    r = score(recommend(p, encoder, members, seen, ids), answers)
                    curve.setdefault(epoch, []).append((r["ndcg"], r["hit"]))
            train(new_model(encoder, s, seed), data, encoder.titles, MAX_EPOCHS, s["lr"], s["decay"], seed, on_epoch=check)
        for epoch, values_ in curve.items():
            ndcg, hit = np.mean(values_, axis=0)
            results.append((hit, ndcg, dict(s, epochs=epoch)))
    results.sort(key=lambda r: (-r[0], -r[1]))
    for hit, ndcg, s in results[:8]:
        print(f"  hit {hit:.4f} ndcg {ndcg:.4f} {s}")
    Path(a.settings).parent.mkdir(parents=True, exist_ok=True)
    Path(a.settings).write_text(json.dumps(results[0][2], indent=2) + "\n")
    print(f"chosen by mean validation hit@{K} over {len(SEEDS)} seeds: {results[0][2]}")


def final(a):
    out = Path(a.out)
    if (out / "results.json").exists():
        raise SystemExit(f"{out / 'results.json'} exists: the holdout was already used")
    rows, holdout, catalog = read_csv(a.train), read_csv(a.holdout), read_csv(a.catalog)
    check_no_holdout(rows, holdout, "event_id")
    settings = json.loads(Path(a.settings).read_text())
    encoder = Encoder(rows, catalog)
    data = examples(rows, encoder)
    answers = {r["member_id"]: r["title_id"] for r in holdout}
    ids = list(answers)
    members, seen = queries(rows, encoder, ids)
    saved = [r["title_id"] for r in read_csv(a.popular)]
    base = {n: score(l, answers) for n, l in baselines(rows, encoder, catalog, ids, seen, saved).items()}

    models = []
    for seed in SEEDS:
        p = train(new_model(encoder, settings, seed), data, encoder.titles, settings["epochs"], settings["lr"],
                  settings["decay"], seed)
        models.append((p, score(recommend(p, encoder, members, seen, ids), answers)))

    model = models[0][1]
    report = {"members": len(ids), "settings": settings,
              "baselines": {n: {"hit@10": r["hit"], "ndcg@10": r["ndcg"], "hits": sum(r["hits"])} for n, r in base.items()},
              "two tower by seed": [{"hit@10": r["hit"], "ndcg@10": r["ndcg"], "hits": sum(r["hits"])} for _, r in models],
              "seed 0 against": {n: dict(zip(["only model hits", "only baseline hits", "p"], mcnemar(model["hits"], r["hits"])))
                                 for n, r in base.items()}}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "two_tower.npz", title_ids=np.array(encoder.ids), **models[0][0])
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    for name, run in (("tune", tune), ("final", final)):
        c = commands.add_parser(name)
        c.set_defaults(run=run)
        c.add_argument("--train", required=True)
        c.add_argument("--catalog", required=True)
        c.add_argument("--settings", default="models/settings.json")
        if name == "final":
            c.add_argument("--holdout", required=True)
            c.add_argument("--popular", required=True, help="the saved top 10")
            c.add_argument("--out", default="models")
    a = p.parse_args()
    # numpy on macOS gives false matmul warnings; loss_and_grads stops on real non-finite values.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        a.run(a)


if __name__ == "__main__":
    main()

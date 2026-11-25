# Viewing data cleaning

A small training project: clean viewing events using a title catalog, then split them by date.
Python 3.9+, no packages to install. The raw files are not in this repository: put them in `data/`.

## Run

```sh
python3 clean_data.py \
  --catalog "data/Netflix title catalog, November 2026.csv" \
  --events "data/Netflix viewing events export, 17 Aug to 16 Nov 2026.csv" \
  --out data/clean --train-end 2026-11-09
python3 -m unittest
```

For another dataset, change the paths and the date. Column names are options too: see `python3 clean_data.py --help`.

## Rules

1. Watch time above the title's catalog runtime: the record is removed.
2. Blank watch time: the record is removed.
3. `completed` is `True` only when the watch time equals the catalog runtime.
4. Training data is every record up to and including 9 Nov 2026; holdout is 10 Nov 2026 onward.

## Result

| File in `data/clean` | Rows | Dates | Completed |
|---|---|---|---|
| `train.csv` | 3,635 | 17 Aug to 9 Nov 2026 | 1 |
| `holdout.csv` | 291 | 10 to 16 Nov 2026 | 0 |

Of 5,000 events, 1,000 were removed for exceeding the runtime and 74 for a blank watch time.
Two columns are added: `completed_raw` (the original value) and `completed_check` (`verified`, or
`no_runtime` when the catalog has no runtime). **Do not use these two columns as model features.**

## Popular top 10

One list for every member: the 10 catalog titles with the most rows in `train.csv`. The holdout is not
read. The catalog file must be in `data/`.

```sh
python3 popular_top10.py --train data/clean/train.csv \
  --catalog "data/Netflix title catalog, November 2026.csv" \
  --out data/baseline/popular_top10.csv --train-end 2026-11-09
```

TT-0412, TT-0347 and TT-0588 have enough views for the list but are not in the catalog, so they are left out.

## Known limits

- Only 1 of the 3,926 kept events equals the catalog runtime exactly, so `completed` is almost constant
  and the holdout has no completed event. A model cannot learn or be tested on this label.
- 278 kept events have no known runtime (175 with titles missing from the catalog, 103 with a blank
  runtime). Rule 1 cannot check them, and they are marked not completed.
- Of the 1,000 events above the runtime, 985 are series. A series runtime may be one episode, in which
  case the removed events may be valid views of several episodes.

The reasons for each rule are in [DECISIONS.md](DECISIONS.md).

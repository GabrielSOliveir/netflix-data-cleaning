# Viewing data cleaning

A small training project: clean viewing events using a title catalog, then split them per member.
Python 3.9+. Cleaning and the top 10 need no packages; the model needs Python 3.10+ and
`pip install -r requirements.txt`. The raw files are not in this repository: put them in `data/`.

## Run

```sh
python3 clean_data.py \
  --catalog "data/Netflix title catalog, November 2026.csv" \
  --events "data/Netflix viewing events export, 17 Aug to 16 Nov 2026.csv" \
  --out data/clean
python3 -m unittest
```

For another dataset, change the paths. Column names are options too: see `python3 clean_data.py --help`.

## Rules

1. Watch time above the title's catalog runtime: the record is removed.
2. Blank watch time: the record is removed.
3. `completed` is `True` only when the watch time equals the catalog runtime.
4. For each member, holdout is the last watch of a catalog title they had not watched before. Train is every
   watch before it. Same timestamp: the higher title id counts as later.
5. Every watch after the member's own holdout watch is removed.
6. Members without one holdout title and one watch before it stay whole in train.

## Result

| File in `data/clean` | Rows | Members | Dates | Completed |
|---|---|---|---|---|
| `train.csv` | 3,140 | 598 | 17 Aug to 15 Nov 2026 | 1 |
| `holdout.csv` | 585 (one per member) | 585 | 25 Aug to 16 Nov 2026 | 0 |

Of 5,000 events, 1,000 were removed for exceeding the runtime and 74 for a blank watch time, which leaves 3,926.
Of those, 201 are after the member's own holdout watch and were removed: 41 repeats of the holdout title and
160 other titles (3,140 + 585 + 201 = 3,926). 13 members have no holdout row; their 18 watches stay in train.
Two columns are added: `completed_raw` (the original value) and `completed_check` (`verified`, or
`no_runtime` when the catalog has no runtime). **Do not use these two columns as model features.**

## Popular top 10

One list for every member: the 10 catalog titles with the most rows in `train.csv`. The holdout
is read only to stop the script if any of its rows is in train. The catalog file must be in `data/`.

```sh
python3 popular_top10.py --train data/clean/train.csv \
  --catalog "data/Netflix title catalog, November 2026.csv" \
  --holdout data/clean/holdout.csv --out data/baseline/popular_top10.csv
```

TT-0412, TT-0347 and TT-0588 have enough views for the list but are not in the catalog, so they are left out.

## Two-tower model

**Finding:** the model beats the popular top 10, but a simple favourite-genre rule beats the model.

Member tower: genres and other catalog features of the titles watched, plus country, profile and device.
Title tower: catalog features plus a small vector per title. Both give 64 numbers. The score is their dot product
plus one popularity bias per title. The brief is in [PROMPT.md](PROMPT.md). Only unwatched catalog titles are recommended.

```sh
python3 train_two_tower.py tune --train data/clean/train.csv \
  --catalog "data/Netflix title catalog, November 2026.csv"
python3 train_two_tower.py final --train data/clean/train.csv --holdout data/clean/holdout.csv \
  --catalog "data/Netflix title catalog, November 2026.csv" --popular data/baseline/popular_top10.csv
```

`tune` uses a validation set cut from train and logs one run per setting to W&B (project `netflix-zero`). It needs
`WANDB_API_KEY` in `.env`. `final` reads the holdout and refuses to run again while `results.json` exists in its
output folder. `models/` is not in git. Holdout result, 585 members:

| List | hit@10 | NDCG@10 | Hits |
|---|---|---|---|
| Popular top 10 | 0.164 | 0.097 | 96 |
| Popular, unwatched | 0.174 | 0.105 | 102 |
| Two-tower, 16 dimensions (seed 0) | 0.581 | 0.296 | 340 |
| **Two-tower, 64 dimensions (seed 0)** | **0.603** | **0.310** | **353** |
| Popular in favourite genre, unwatched (no learning) | 0.641 | 0.337 | 375 |

The 64-dimension model beats both popular lists (McNemar p < 0.001) and **loses to the genre rule** (p = 0.017):
almost all of its gain is the member's favourite genre. Over seeds 0 to 4 it gives 0.573 to 0.603 (mean 0.587) and
the 16-dimension model 0.568 to 0.600 (mean 0.585), so 64 is not clearly better than 16. The 64-dimension score is a
second look at the holdout: 64 comes from the brief, and its other settings were chosen on validation.

## Known limits

- Each member is cut at their own holdout watch, not at one date. Train, and so the top 10, includes watches by
  other members that are later than a given member's holdout watch.
- Only 1 of the 3,926 kept events equals the catalog runtime exactly, so `completed` is almost constant
  and the holdout has no completed event. A model cannot learn or be tested on this label.
- 278 kept events have no known runtime (175 with titles missing from the catalog, 103 with a blank
  runtime). Rule 1 cannot check them, and they are marked not completed.
- Of the 1,000 events above the runtime, 985 are series. A series runtime may be one episode, in which
  case the removed events may be valid views of several episodes.

The reasons for each rule are in [DECISIONS.md](DECISIONS.md).

## Top Picks API

With Python 3.11, run `pip install -r requirements-api.txt` and
`uvicorn api:app`. Request `GET /recommendations/M-10073` or open `/docs`.
At startup the service upserts 300 catalog titles into embedded Qdrant. For a
persistent Qdrant server, set `QDRANT_URL` to its URL and use one startup worker.

The serving model combines the trained member and title towers with a
64-dimensional TF-IDF/latent semantic description channel. A member's text
vector averages the descriptions of catalog titles they played. Retrieval uses
cosine similarity over the two channels, with a Qdrant exclusion filter for
every catalog title played in train. Item-specific popularity and ID offsets
are omitted for all titles so a new release is scored on the same content basis.
The previous training metrics apply to the original scorer.

The serving export contains the title vectors, catalog response fields, and
member-tower inference weights. To rebuild it using the original inputs, run
`python build_index_data.py --catalog "data/Netflix title catalog, November 2026.csv" --checkpoint models/model_64/two_tower.npz`.
Run `SAVE_RESPONSES=1 python -m unittest -v test_api` to validate the contract
and regenerate both endpoint samples in `saved_responses/`.

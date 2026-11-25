Build a small two-tower recommender that has to beat the popular top 10.

Data: data/clean/train.csv and data/clean/holdout.csv, and the November 2026 catalog. Holdout is a member's last new catalog title; 585 of 598 members have one, and the others stay in train only.

Towers (one hidden layer each):
- Title tower: catalog features (genres, title type, maturity rating, runtime, release year) plus a learned 64-dimension vector per title.
- Member tower: the average features of the catalog titles the member watched, plus country, profile type, device and the number of watches.
- Both output 64-dimension vectors. Score = dot product plus one bias per title, which starts at log popularity.

Training rounds: validation is a member's last new catalog title in train (549 members). Tune hidden size (16, 32, 64), learning rate (0.003, 0.01), decay (0.1, 0.5, 1, 2) and epochs (up to 60, checked every 5) with 5 seeds each. Pick the setting and epoch with the best mean validation hit@10; NDCG@10 breaks ties. Log one run per setting (24 runs, mean of the 5 seeds) to W&B project "netflix-zero".

Check: the model recommends only unwatched catalog titles. Score hit@10 and NDCG@10 on holdout for seeds 0 to 4 (seed 0 is the model), next to the saved popular top 10 (which does not skip watched titles), popular unwatched titles and the favourite-genre rule (unwatched titles). The holdout has been scored in two final runs: 16 dimensions, then 64.

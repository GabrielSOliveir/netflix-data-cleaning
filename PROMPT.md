Build a small two-tower recommender that has to beat the popular top 10.

Data: data/clean/train.csv and data/clean/holdout.csv (per member: holdout is each member's last new catalog title), and the November 2026 catalog.

Towers:
- Title tower: catalog features (genres, title type, maturity rating, runtime, release year) plus a small learned vector per title.
- Member tower: the average features of the titles the member watched, plus country, profile type and device.
- Both output 64-dimension vectors. Score = dot product.

Training rounds: validation is each member's last new catalog title in train. Tune hidden size, learning rate and decay with 5 seeds each, up to 60 epochs, checking every 5. Pick the setting with the best mean validation hit@10. Log each run to W&B project "netflix-zero".

Check: recommend only unwatched catalog titles. Score hit@10 on holdout once, next to the popular top 10 and the favourite-genre rule.

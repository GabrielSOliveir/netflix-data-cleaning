# Decisions

1. **Remove events above the catalog runtime** (1,000 of 5,000). The whole record goes; no value is capped.
2. **Remove events with a blank watch time** (74). Filling them would invent a full view.
3. **Completed means watch time equals the catalog runtime exactly.** The original indicator is not used.
   It is kept in `completed_raw` for auditing.
4. **Keep events with no known runtime** (278). No rule says to remove them, but a match cannot be
   proved, so they are not completed. They are marked `no_runtime` in `completed_check`.
5. **Split by date after cleaning.** Train ends on 9 Nov 2026 (included); holdout starts on 10 Nov 2026.
6. **Do not publish the raw files.** Only the two cleaned datasets are in this repository.
7. **Standard library only**, so the script runs without a virtual environment.
8. **The popular top 10 counts rows in the clean training data.** Ties go to the lower title id. The script
   stops if a row is later than the training end date.
9. **Only recommend titles that exist in the catalog.** This applies to both the baseline (the popular top 10)
   and the model. It removed TT-0412, TT-0347 and TT-0588 from the top 10; TT-0129, TT-0004 and TT-0054 took
   their places. A title missing from the catalog has no name to show, and one shared rule keeps the
   comparison between baseline and model fair.

# Decisions

1. **Remove events above the catalog runtime** (1,000 of 5,000). The whole record goes; no value is capped.
2. **Remove events with a blank watch time** (74). Filling them would invent a full view.
3. **Completed means watch time equals the catalog runtime exactly.** The original indicator is not used.
   It is kept in `completed_raw` for auditing.
4. **Keep events with no known runtime** (278). No rule says to remove them, but a match cannot be
   proved, so they are not completed. They are marked `no_runtime` in `completed_check`.
5. **Split per member after cleaning** (replaces the split by date, 9/10 Nov 2026). Holdout is the member's last
   watch of a catalog title they had not watched before; train is every watch before it.
   - Tie: watches with the same timestamp are ordered by title id, so the higher id counts as later. Dates
     have no time of day, so every same-day watch is a tie. If the title id is also equal, file order decides.
   - Every watch after the member's own holdout watch is removed, using the same order: 201 rows, of which 41
     repeat the holdout title and 160 are other titles. Otherwise train would show the member's future.
   - Members without one holdout title and one watch before it stay whole in train (13 members, 18 rows).
   - Accepted: each member is cut at their own holdout watch, not at one date. Train still has watches by
     other members that are later than a given holdout watch.
6. **Do not publish the raw files.** Only the two cleaned datasets are in this repository.
7. **Standard library only**, so the script runs without a virtual environment.
8. **The popular top 10 counts rows in the clean training data.** Ties go to the lower title id. The script
   stops if any row (by event id) is also in the holdout. This replaces the check by date (9 Nov 2026).
9. **Only recommend titles that exist in the catalog.** This applies to both the baseline (the popular top 10)
   and the model. It keeps TT-0412, TT-0347 and TT-0588 out of the top 10. A title missing from the catalog has no name to show, and one shared rule keeps the
   comparison between baseline and model fair.
10. **No code or check copies the project folder or touches `.env`.** Checks that need a copy copy named files
    only. `test_no_secret_copies.py` enforces this for the Python files in the repository.

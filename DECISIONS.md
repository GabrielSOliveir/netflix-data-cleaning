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
11. **The two-tower model uses numpy** (pinned in `requirements.txt`); cleaning, split and top 10 stay on the
    standard library. Alternative: PyTorch, too heavy for 300 titles and about 3,000 rows.
12. **How the model is compared** (written before any training).
    - Main metric: hit@10, the share of holdout members whose holdout title is in their 10. Second: NDCG@10.
    - Two baselines: the saved top 10 as it is, and popular titles the member has not watched. The holdout is
      always an unwatched title and the model skips watched ones, so only the second isolates personalisation.
    - The model beats a baseline if hit@10 is higher and the paired exact McNemar test gives p < 0.05.
    - Settings are chosen on a validation set cut from train with the same split per member. The holdout is
      read once, by the final run, after the settings are fixed. Seeds 0 to 4 are all reported; seed 0 is the model.
    - Only `title_id` of a holdout row is used, as the answer. Member features come from their train rows.
    - Added after validation and before the holdout was read: a third reference with no learning, popular
      unwatched titles in the member's favourite genre. It scored above the first model on validation, so a win
      over the popular lists alone does not show that the two towers are needed.
13. **Result of the single holdout run:** the two-tower model beats the popular top 10 (340 hits against 96 of 585)
    and loses to the genre rule (375). The holdout is now used: a new model is chosen on validation, and a
    second holdout score must be reported as a second look.

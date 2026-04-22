# Martin — work log (Recipe Retrieval & Ranking)

**Last updated:** 2026-04-22 (UTC)

## Role (proposal)

Martin owns **Recipe Retrieval and Ranking** for the vision-based recipe system: take a list of detected ingredients with confidence scores, optionally map them to a shared vocabulary, score and rank recipe candidates, and support evaluation (Top‑k hit rate, MRR) so the team can show measurable progress. This role sits between the vision model output and the end-to-end demo; integration should consume a small, stable API rather than re‑implementing scoring ad hoc.

## Where we are

The `recipe_retrieval` package is implemented end to end: corpus loading, an inverted index over recipe ingredients, three baseline rankers (overlap, confidence‑weighted, penalty for missing recipe ingredients), a `retrieve` entry point matching the `test.py` list‑of‑dicts shape, a CLI for one‑off demos and ablation eval with JSON artifacts, optional alias‑based normalization for when the team freezes vocabulary, and unit tests on sample data. Documentation is in the main `README` under “Recipe retrieval and ranking (Martin)”. A separate git branch (see below) holds these commits so `main` can stay untouched until the group is ready to merge.

## What is left (and why)

- **Load and index the real team recipe corpus (e.g. Recipe1M+ slice or agreed export):** not done here because the shared cleaned dataset path and format were not the retrieval owner’s to finalize alone; when available, point `build_index_from_paths` at those files (or add a one‑row loader if the schema differs slightly).
- **Lock detector output and confidence policy:** still owned by the vision teammate; ranker weight tuning and eval should be re‑run after thresholds and class names stabilize.
- **Shared ingredient vocabulary / alias map as the single source of truth:** partially supported via `AliasFileNormalizer` and `retrieve_with_reconciled_vocab`; a committed `data/team_ingredient_alias.json` (or team‑agreed name) is pending the normalization sub‑owner.
- **End‑to‑end demo wiring (image → detect → rank):** owned by integration; this repo provides `retrieve` and the CLI—hook `test.py` (or local YOLO inference) into `retrieve` on the feature branch or after merge.
- **Stronger IR baselines (BM25, embeddings):** not implemented yet; current deliverable matches the project plan’s rule‑based baselines; add if the instructor expects extra depth and time allows.

**Keeping a personal branch in sync with `main`:** after pushing your branch, run `git fetch origin` and either `git merge origin/main` or `git rebase origin/main` on your branch whenever others land work on `main`. Resolve conflicts, run tests, then push. That keeps your GitHub copy current without changing `main` until you open a PR.

**Suggested branch name for Martin’s work:** `martin/recipe-retrieval` (or similar).

---

## Chronological deliverables (implemented)

| Date (UTC) | What | File(s) / location |
|------------|------|--------------------|
| 2026-04-22 | Core types: detections, normalized ingredients, ranked recipes, score breakdown, `RetrievalResult` | `recipe_retrieval/schema.py` |
| 2026-04-22 | Normalizers: identity; JSON alias file for post–vocab-freeze | `recipe_retrieval/normalize.py` |
| 2026-04-22 | Recipe corpus: JSON/JSONL/array loader, `RecipeRecord` | `recipe_retrieval/corpus.py` |
| 2026-04-22 | Text keys for matching | `recipe_retrieval/text.py` |
| 2026-04-22 | Inverted index over ingredient keys | `recipe_retrieval/index.py` |
| 2026-04-22 | Rankers: overlap, confidence‑weighted, penalty‑aware + `PenaltyConfig` | `recipe_retrieval/rankers.py` |
| 2026-04-22 | Pipeline: `build_index_from_paths`, `retrieve` | `recipe_retrieval/pipeline.py` |
| 2026-04-22 | Eval: cases, ablation, Top‑1/3/5 + MRR, JSON artifacts | `recipe_retrieval/eval.py` |
| 2026-04-22 | CLI: `demo` and `eval` subcommands | `recipe_retrieval/cli.py` |
| 2026-04-22 | Post‑freeze integration helper | `recipe_retrieval/integrate.py` |
| 2026-04-22 | Public exports | `recipe_retrieval/__init__.py` |
| 2026-04-22 | Sample recipe corpus, eval cases, query, optional alias map | `data/sample_recipes.jsonl`, `data/eval_cases.jsonl`, `data/sample_query.json`, `data/sample_alias_map.json` |
| 2026-04-22 | Unit tests | `tests/test_retrieval.py` |
| 2026-04-22 | User‑facing how‑to (retrieval + vision notes) | `README.md` |
| 2026-04-22 | Ignore eval run outputs and bytecode | `.gitignore` |
| 2026-04-22 | This living progress / status report | `Martin-work-done.md` |

---

*Update this file when you add features, re‑run major evals, or when dependencies (detector, vocabulary, full recipe data, demo) change status.*

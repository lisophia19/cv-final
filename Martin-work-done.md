# Martin — Retrieval & ranking: status and deliverables

**Last updated:** 2026-04-22 (UTC)  
**Owner:** Martin (Recipe Retrieval & Ranking)  
**Purpose of this document:** A single, team-readable source of truth for what the retrieval layer does, what is already built, what is still open, and who is needed for each dependency. It is maintained as the project evolves; treat it as the primary orientation doc for this workstream’s scope.

---

## At a glance

| Item | Status |
|------|--------|
| **Scope** | Turn detected ingredients (with confidences) into a ranked list of recipe suggestions; support evaluation and a stable hook for the full pipeline. |
| **Implementation** | `recipe_retrieval` package: indexing, three ranking strategies, eval + CLI, tests, sample data. **Shipped** on branch `martin/recipe-retrieval` (not required to be on `main` yet). |
| **Blocking others?** | No: demo/integration can start whenever they are ready; the agreed input shape matches `test.py` post-processing. |
| **Waiting on** | Full recipe dataset format/path from data owners; final ingredient vocabulary/aliases from normalization; stable detector output from vision. |

---

## How this fits the full product (for everyone)

The class pipeline is: **photo → list of ingredients → match to recipes → show ranked recipes (and a demo).** Martin’s work is the middle two steps: take the ingredient list the vision system produces, match it against our recipe database, and order recipes so the ones that best use what the user has (and that we are confident about) appear first. Nothing here trains the camera model; that stays with the vision track. The integration teammate wires “image in” to the detector and “ranked list out” to the UI; Martin supplies the ranking behavior and the evaluation tooling so the team can show evidence of quality, not just a demo.

**Plain-language input we expect from upstream:** a list of objects like `{ "ingredient": "<name from model>", "confidence": 0.0–1.0 }`, the same idea as the list built in `test.py` after deduplication.

**Plain-language output for downstream:** a ranked top‑k of recipes, each with an ID, title, a single score, and a short breakdown of how that score was built (so debugging and the write-up are easier).

---

## Martin’s responsibilities (contract)

| Direction | What we own |
|-----------|------------|
| **In** | Detected ingredients + confidences; optional team alias map so names from the model line up with names in the recipe data. |
| **Out** | Ranked recipes + scores; reproducible eval runs (e.g. “Does the right recipe appear in the top 1, 3, or 5?”) and simple CLI to demo or benchmark without writing new scripts each time. |
| **We do not own** | Training or maintaining the YOLO/Roboflow model; the full app or UI; curating the production Recipe1M+ (or other) export unless the team re-assigns that. |

---

## Current state (where we are)

The retrieval track is in a **working, test-covered baseline** on branch **`martin/recipe-retrieval`**. The code loads recipe files (JSON/JSONL), builds a search index over recipe ingredients, scores each candidate with one of three strategies (simple overlap; overlap weighted by how confident the detector is; same plus a penalty for recipes that need many ingredients we did not see), and exposes a small Python API plus a command-line tool for a one-off demo and for running ablation comparisons and saving JSON result files. Documentation for how to run this lives in the project **`README`**, section **“Recipe retrieval and ranking (Martin).”** Sample recipes and test queries under `data/` allow anyone to run the pipeline without private datasets.

**Why `main` may not have this yet:** the branch is intentionally off `main` so Martin can push to GitHub and iterate without forcing a merge on teammates who are not depending on it yet. Merging to `main` is a team decision, usually via pull request when integration is ready.

---

## What remains, by theme (and why it is not done yet)

| Theme | What | Owner / reason |
|--------|------|----------------|
| **Data** | Point the indexer at the **real** team recipe export (path, schema, and any cleaning rules). | Blocked on agreed corpus and, often, the member(s) who own Recipe1M+ prep or a smaller agreed slice. |
| **Vocabulary** | A single **alias map** (e.g. “tomatoes” → “tomato”) committed to the repo so detector strings and recipe strings align. | Blocked on normalization/vocabulary work; the code already supports a JSON map when the file exists (`AliasFileNormalizer` / `retrieve_with_reconciled_vocab`). |
| **Vision** | **Stable** detection: confidence thresholds, class names, and contract for what gets sent to retrieval. | Blocked on vision; we will re-tune and re-run evals after the model side stabilizes. |
| **End-to-end** | One script or app path: **image → detect → rank → show.** | Integration/demo owner; retrieval exposes `retrieve` and the CLI to plug in. |
| **Stretch (optional)** | Stronger search baselines (e.g. text similarity beyond rule scores, ANN) if the course or timeline asks for it. | Not in the first baseline by design; add only if we have time and a clear grading benefit. |

---

## Risks and mitigations (brief)

- **Name mismatch** between YOLO classes and recipe text remains the #1 product risk; mitigation is a shared alias list + one person accountable for it.
- **Eval on toy data** only measures pipeline correctness, not final quality; mitigation is to re-point eval at the same recipe split the team uses for reporting, once that exists.
- **Branch drift** from `main`; mitigation is to merge or rebase `main` into `martin/recipe-retrieval` regularly (see `README` / git workflow) before big integration milestones.

---

## How to get oriented in the code (5 minutes)

1. Read **`README.md`** — “Recipe retrieval and ranking (Martin).”  
2. Open **`recipe_retrieval/pipeline.py`** — `retrieve` is the main integration call.  
3. Run **tests:** `python3 -m unittest discover -s tests -v` from the repo root.  
4. Run a **sample demo:** `python3 -m recipe_retrieval.cli demo --recipes data/sample_recipes.jsonl --query data/sample_query.json`

---

## Deliverables log (implemented work)

*Each line is a concrete increment. New rows should be appended when the workstream changes material.*

| Date (UTC) | Deliverable | Location |
|------------|-------------|----------|
| 2026-04-22 | Data structures for query, normalized tokens, ranked results, and score details | `recipe_retrieval/schema.py` |
| 2026-04-22 | Normalization: pass-through; optional JSON alias file for post–vocab-freeze | `recipe_retrieval/normalize.py` |
| 2026-04-22 | Recipe file loading (JSON/JSONL) and in-memory `RecipeRecord` | `recipe_retrieval/corpus.py` |
| 2026-04-22 | Consistent string matching for ingredients | `recipe_retrieval/text.py` |
| 2026-04-22 | Inverted index (ingredient → recipes) for candidate lookup | `recipe_retrieval/index.py` |
| 2026-04-22 | Three rankers: overlap; confidence-weighted; penalty for missing recipe ingredients | `recipe_retrieval/rankers.py` |
| 2026-04-22 | `build_index_from_paths` + `retrieve` (main entry for integration) | `recipe_retrieval/pipeline.py` |
| 2026-04-22 | Eval harness: per-case and aggregate metrics; ablation; JSON artifacts under `runs/retrieval_eval/` when using CLI | `recipe_retrieval/eval.py` |
| 2026-04-22 | CLI: `demo` (try a query) and `eval` (benchmark rankers) | `recipe_retrieval/cli.py` |
| 2026-04-22 | Post-freeze integration helper using a team alias file | `recipe_retrieval/integrate.py` |
| 2026-04-22 | Public package surface | `recipe_retrieval/__init__.py` |
| 2026-04-22 | Sample recipes, eval cases, example query, sample alias file | `data/sample_recipes.jsonl`, `data/eval_cases.jsonl`, `data/sample_query.json`, `data/sample_alias_map.json` |
| 2026-04-22 | Automated tests | `tests/test_retrieval.py` |
| 2026-04-22 | User-facing run instructions (retrieval + related notes) | `README.md` |
| 2026-04-22 | Git ignores for local eval output and caches | `.gitignore` |
| 2026-04-22 | This status and deliverables log | `Martin-work-done.md` |

---

## Maintaining this document (process)

**Who updates it:** Martin (or a delegate) whenever a deliverable ships, a dependency clears or stalls, or an integration milestone changes what others need to know.  
**What to add:** A new row in the table above (date = merge or “feature complete” day), a short change to *Current state* or *What remains* if scope shifts, and a bump to *Last updated* at the top.  
**Tone:** Factual, concise, and written so a new teammate (or a grader) can understand scope and blockers in one read.

If something in this file disagrees with the `README` or the code, **code and README win** until the doc is updated.

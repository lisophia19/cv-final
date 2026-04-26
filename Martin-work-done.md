# My retrieval & ranking work — status and deliverables

**Last updated:** 2026-04-26 (UTC, post Sprint B eval pass)  
**Who I am on this project:** Martin (Recipe Retrieval & Ranking)

**Why I keep this file:** I want anyone on the team (or grading the project) to open one document and understand what I am responsible for, what I have already shipped, what is still open, and what I am waiting on from others. I will update it as my workstream changes.

---

## At a glance

| Item | Status |
|------|--------|
| **What I own** | Turning detected ingredients (with confidences) into a ranked list of recipe suggestions; evaluation tooling; a stable hook for whoever does end-to-end integration. |
| **What I shipped** | The `recipe_retrieval` package: indexing, three ranking strategies, eval + CLI, tests, sample data, and app wiring so `app.py` now calls retrieval instead of returning a stub. I also verified the app behavior end-to-end (upload -> detections -> suggested recipes) in the current environment. It lives on my branch **`martin/recipe-retrieval`** (it does not have to be on `main` yet). |
| **Am I blocking you?** | I do not believe so. If you are doing the demo, you can call into my API whenever you are ready. The input shape I expect matches the list-of-dicts idea from `test.py` after deduplication. |
| **What I am waiting on** | A real recipe export (path + format) from whoever owns the dataset; a finalized ingredient vocabulary / alias map from normalization; stable detector output (names + thresholds) from vision. None of these block a semi-running integrated pipeline at this point. |

---

## How my piece fits the full product

Our pipeline is: **photo → list of ingredients → match to recipes → show ranked recipes (and a demo).** I own the middle two steps: I take the ingredient list the vision side produces, match it against our recipe data, and rank recipes so the ones that best use what the user appears to have (and that I can trust based on confidence) float to the top. I am **not** training or maintaining the camera model; that stays with whoever owns vision. Whoever owns integration wires “image in” to the detector and “ranked list out” to the UI; I supply the ranking logic and the evaluation tooling so we can show evidence of quality, not only a one-off demo.

**What I expect from upstream (plain language):** a list like `{ "ingredient": "<name from the model>", "confidence": <number> }` per detected item, same spirit as the list built in `test.py` after deduplication.

**What I hand downstream:** a ranked top‑k of recipes, each with an ID, title, one overall score, and a short breakdown of how I computed that score (so debugging and the write-up are easier).

---

## My contract (what I take in and what I give back)

| Direction | What I own |
|-----------|------------|
| **In** | Detected ingredients + confidences; optionally a team alias map so model names line up with recipe ingredient text. |
| **Out** | Ranked recipes + scores; reproducible eval runs (for example: did the “right” recipe land in the top 1, 3, or 5?); a small CLI so nobody has to re-write glue scripts to try a query or run an ablation. |
| **Out of scope for me** | Training or maintaining YOLO / Roboflow; the full app or UI; curating the production Recipe1M+ export unless the team explicitly re-assigns that to me. |

---

## Where I am right now

I have a **working, test-covered baseline** on **`martin/recipe-retrieval`**. My code loads recipe files (JSON/JSONL), builds a search index over recipe ingredients, scores candidates with one of three strategies I implemented (simple overlap; overlap weighted by detector confidence; overlap weighted plus a penalty when the recipe still needs ingredients I did not see), and exposes a small Python API plus a CLI for a quick demo and for ablation runs that write JSON under `runs/retrieval_eval/` when you use the eval command. I documented how to run everything in the project **`README`**, under **“Recipe retrieval and ranking (Martin).”** I also added sample recipes and queries under `fridge_data/` so anyone can exercise the pipeline without my private files. As of 2026-04-26, I wired the Streamlit demo (`app.py`) to call this retrieval module, so the demo path is no longer a retrieval stub.

**Why this might not be on `main` yet:** I kept my work on a separate branch so I can push to GitHub and iterate without forcing a merge on teammates who told me they are not depending on my code yet. When we are ready, I will open a PR (or whatever process we agree on) to merge into `main`.

---

## What I still have to do (and why I have not done it yet)

| Theme | What is left | Why |
|--------|--------------|-----|
| **Data** | Point my indexer at our **real** recipe export (exact path, schema, cleaning rules). | I am waiting on whoever owns the Recipe1M+ slice (or our agreed substitute) to give me a stable artifact I should not invent on my own. |
| **Vocabulary** | A single **alias map** in the repo (for example “tomatoes” → “tomato”) so detector strings and recipe strings line up. | I am waiting on normalization / shared vocabulary; I already wired support for a JSON alias file when you have one (`AliasFileNormalizer` / `retrieve_with_reconciled_vocab`). |
| **Vision** | **Stable** detection: thresholds, class names, and a clear contract for what gets sent to me. | I am blocked on vision stabilizing; once it does, I will re-tune ranker weights if needed and re-run evals on our agreed set. |
| **End-to-end** | Keep validating the semi-running path **image → detect → rank → show** in the target demo environment. | I completed retrieval wiring in `app.py` and verified that upload now updates detections and suggestions. Remaining work is quality tuning with final data/vocabulary. |
| **Stretch** | Stronger retrieval baselines (for example BM25 or embeddings) if we want extra depth. | I did not prioritize it for the first baseline; I will add it only if the timeline and grading expectations make it worth it. |

---

## Retrieval quality sprint plan (my next focused work)

I am organizing my next work into short, reviewable chunks so quality improvements are clear and measurable.

1. **Sprint A — contract hardening and robustness (high confidence, immediate)**
   - Add/expand tests for edge inputs (empty detections, duplicate ingredients, low confidence only, unknown tokens).
   - Ensure retrieval output ordering and tie-breaking are deterministic.
   - Add clearer error messages for bad recipe files or malformed records.
   - **Definition of done:** tests pass; no regression in existing retrieval behavior.

2. **Sprint B — retrieval evaluation quality pass (high value, immediate)**
   - Expand eval cases beyond toy happy-path examples.
   - Run all baseline rankers against the same case set and store artifacts.
   - Capture a compact before/after comparison note for tuning changes.
   - **Definition of done:** repeatable eval command + saved artifact + short interpretation.
   - **Status update (2026-04-26):** completed on an expanded case file (`fridge_data/eval_cases_sprint_b.jsonl`) and artifact logged.

3. **Sprint C — scoring tuning after dependencies freeze (blocked on others)**
   - Tune weighting/penalties after the team finalizes:
     - recipe dataset format/path,
     - alias map,
     - detector naming/threshold behavior.
   - **Definition of done:** stable tuned defaults with measured improvement on agreed eval split.

---

## Retrieval benchmark snapshot

This section tracks retrieval-side measured performance only (my role), so progress is objective.

### Latest verified run
- **Date:** 2026-04-26
- **Command:** `python3 -m recipe_retrieval.cli eval --recipes fridge_data/sample_recipes.jsonl --cases fridge_data/eval_cases_sprint_b.jsonl --out runs/retrieval_eval`
- **Result summary:**
  - `overlap`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
  - `confidence_weighted`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
  - `penalty_aware`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
- **Interpretation:** plumbing is correct on the expanded sample set, and all rankers are stable. Because the recipe corpus is still very small and curated, this is still not sufficient evidence of final real-world ranking quality.

### Quality caveat
- Current metrics are from sample data meant for integration reliability.
- Final benchmark should be recorded after dataset/vocabulary/detector output are stabilized by the relevant owners.
- I should expect metric spread between rankers only after moving to a larger, noisier evaluation set.

---

## Team role deliverables status (cross-team view)

This section captures team-level deliverables and my current understanding of status/dependencies so planning is explicit.

### Group Members 1 & 3 — Ingredient Detection (Vision Model)
- **Deliverable:** image -> clean ingredient list with confidence scores.
  - **Status:** **Partially done** (pipeline produces detections; quality still evolving).
- **Deliverable:** detection accuracy ownership and iterative model improvements.
  - **Status:** **In progress**.
- **Depends on already done:** ingredient dataset, training/inference scripts.

### Group Members 1 & 3 — Data Processing and Normalization
- **Deliverable:** cleaned, query-ready recipe dataset with stable schema/path.
  - **Status:** **Not finalized**.
- **Deliverable:** shared ingredient vocabulary + normalization map across modules.
  - **Status:** **Not finalized** (my retrieval supports alias-map ingestion when provided).
- **Depends on already done:** team agreement on canonical ingredient naming and dataset source.

### Group Member 2 (me) — Recipe Retrieval and Ranking
- **Deliverable:** ranked top-k recipe retrieval from detected ingredient list.
  - **Status:** **Done (baseline integrated and verified)**.
- **Deliverable:** multiple ranking strategies + scoring logic tied to overlap/confidence/missing ingredients.
  - **Status:** **Done**.
- **Deliverable:** retrieval-side eval and ranker comparison.
  - **Status:** **Done for baseline/sample set; final pass pending final dataset/vocab/detector freeze**.
- **Depends on already done:** detection output contract and retrieval integration hook (available).

### Group Member 4 — End-to-End Integration and Demo
- **Deliverable:** stitched pipeline (image in -> ranked recipes out) with demo interface.
  - **Status:** **Partially done** (semi-running path now works in app).
- **Deliverable:** final demo polish and presentation flow.
  - **Status:** **In progress**.
- **Depends on already done:** stable module interfaces from detection + retrieval + normalization.

---

## Ordered deliverables by owner

1. **Ingredient detection output contract (ingredient + confidence)** — **Group Members 1 & 3**
2. **Recipe dataset prepared in agreed schema/path** — **Group Members 1 & 3**
3. **Shared ingredient alias/vocabulary mapping** — **Group Members 1 & 3**
4. **Recipe retrieval/ranking engine (top-k + score breakdown)** — **Group Member 2 (me)**
5. **Retrieval evaluation and ranker benchmark artifacts** — **Group Member 2 (me)**
6. **End-to-end integration wiring (upload -> detect -> rank -> display)** — **Group Member 4**
7. **End-to-end validation + demo polish for presentation** — **Group Member 4**
8. **Final retrieval tuning pass after dependency freeze** — **Group Member 2 (me), dependent on items 2-3 and detector stability**

---

## Risks I am watching (and what I am doing about them)

- **Name mismatch** between model classes and recipe text is still the biggest product risk. I am counting on a shared alias list and a clear owner for vocabulary so I am not guessing in silo.
- **Eval on sample data** only proves the plumbing, not final quality. As soon as we have a team eval set on real recipes, I will point my harness at it.
- **Branch drift** from `main`. I plan to merge or rebase `main` into `martin/recipe-retrieval` regularly so integration is not painful later.

---

## How to get oriented in my code (about five minutes)

1. Read **`README.md`** — section **“Recipe retrieval and ranking (Martin).”**  
2. Open **`recipe_retrieval/pipeline.py`** — `retrieve` is the main call for integration.  
3. From the repo root, run **`python3 -m unittest discover -s tests -v`**.  
4. Try a sample run: **`python3 -m recipe_retrieval.cli demo --recipes fridge_data/sample_recipes.jsonl --query fridge_data/sample_query.json`**

---

## Deliverables log (what I implemented)

*I append a row here when I ship something material.*

| Date (UTC) | What I delivered | Where |
|------------|------------------|-------|
| 2026-04-22 | Types for my query, normalized tokens, ranked results, and score breakdown | `recipe_retrieval/schema.py` |
| 2026-04-22 | Normalization: pass-through; optional JSON alias file for after vocabulary freeze | `recipe_retrieval/normalize.py` |
| 2026-04-22 | Recipe loading (JSON/JSONL) and `RecipeRecord` | `recipe_retrieval/corpus.py` |
| 2026-04-22 | Consistent ingredient string keys for matching | `recipe_retrieval/text.py` |
| 2026-04-22 | Inverted index (ingredient → recipes) | `recipe_retrieval/index.py` |
| 2026-04-22 | Three rankers: overlap; confidence-weighted; penalty for missing recipe ingredients | `recipe_retrieval/rankers.py` |
| 2026-04-22 | `build_index_from_paths` + `retrieve` (integration entry point) | `recipe_retrieval/pipeline.py` |
| 2026-04-22 | Eval harness, ablations, JSON artifacts when using the CLI | `recipe_retrieval/eval.py` |
| 2026-04-22 | CLI: `demo` and `eval` | `recipe_retrieval/cli.py` |
| 2026-04-22 | Post-freeze integration helper for a team alias file | `recipe_retrieval/integrate.py` |
| 2026-04-22 | Public exports for the package | `recipe_retrieval/__init__.py` |
| 2026-04-22 | Sample recipes, eval cases, example query, sample alias file | `fridge_data/sample_recipes.jsonl`, `fridge_data/eval_cases.jsonl`, `fridge_data/sample_query.json`, `fridge_data/sample_alias_map.json` |
| 2026-04-26 | Synced my branch with latest `main` and reconciled path changes from `data/` to `fridge_data/` for retrieval samples and docs | `README.md`, `tests/test_retrieval.py`, `recipe_retrieval/cli.py`, `recipe_retrieval/integrate.py`, `Martin-work-done.md` |
| 2026-04-26 | Wired Streamlit demo retrieval path (`get_recipes`) to call the real ranking pipeline with sample corpus + alias fallback | `app.py` |
| 2026-04-26 | Verified integration quality gate: unit tests pass, retrieval eval CLI runs, and app behavior shows detections + suggested recipes after upload | `tests/test_retrieval.py`, `recipe_retrieval/cli.py`, `app.py`, `Martin-work-done.md` |
| 2026-04-26 | Sprint B evaluation pass: expanded retrieval eval case set and benchmark rerun artifact | `fridge_data/eval_cases_sprint_b.jsonl`, `runs/retrieval_eval/retrieval_eval_20260426T190805Z.json`, `Martin-work-done.md` |
| 2026-04-22 | Unit tests | `tests/test_retrieval.py` |
| 2026-04-22 | README instructions for retrieval (and related notes) | `README.md` |
| 2026-04-22 | Gitignore updates for local eval output and bytecode | `.gitignore` |
| 2026-04-22 | This living status log | `Martin-work-done.md` |

---

## How I maintain this document

When I ship a meaningful change, I add a row to the table above, bump **Last updated**, and adjust **Where I am** or **What I still have to do** if my scope or blockers changed. If this file ever disagrees with the `README` or the code, **I treat the README and code as correct** until I fix the doc.

# My retrieval & ranking work — status and deliverables

**Last updated:** 2026-05-04 (UTC, post-merge poster figure + YOLO path fix)  
**Who I am on this project:** Martin (Recipe Retrieval & Ranking)

---

## Executive status

- **Current state:** retrieval/ranking is implemented, tested, benchmarked, and integrated in `app.py`.
- **Readiness:** semi-running pipeline is working (`upload -> detect -> suggested recipes`).
- **Main blockers:** final recipe dataset format/path, shared alias map, stabilized detector naming/thresholds.
- **Immediate next action:** run final tuning and benchmark pass as soon as those blockers are resolved.
- **Owner:** Martin (retrieval/ranking), with dependencies on Group Members 1 & 3.

---

## Current readiness and blockers

### Ready now
- Retrieval module consumes ingredient detections and returns ranked top-k recipes.
- Three rankers are implemented and selectable.
- Retrieval eval and artifact logging run from CLI.
- App calls real retrieval (no retrieval stub).

### Waiting on dependencies
- Production recipe corpus in agreed schema/path.
- Team-owned canonical ingredient alias map.
- Stable detector output contract for final tuning baseline.

---

## What changed since last update

- Merged latest `main` (team moved YOLO weights under `yolo/`); `eval_e2e.py` now resolves `yolo/.../best.pt` first, then legacy `runs/...`.
- Added `scripts/plot_poster_topk.py` and default output `figures/poster_topk_retrieval.png` for the poster’s top-k retrieval bar chart (reads latest `runs/e2e_eval/e2e_eval_*.json` unless `--json` is passed).
- Rehauled this document for team-facing hybrid format (status first, technical evidence second).
- Completed Sprint A hardening (input validation, deterministic ordering, edge-case tests).
- Completed Sprint B evaluation pass with expanded case set.
- Added Sprint C prep (config-driven tuning surface for quick post-freeze experiments).
- Synced latest `main`, then ran E2E baseline on team eval set (`eval_data/labels.jsonl`).
- Added tunable E2E eval controls in `eval_e2e.py` and completed one controlled tuning pass.
- Added failure-pattern summary to drive concrete team handoffs.

---

## Retrieval strategy summary (plain English)

All strategies are fully implemented and working as expected on current sample evaluation data.

1. **Overlap**
   - Scores recipes by how many detected ingredients match.
   - Simplest baseline; ignores confidence strength.
   - **Status:** implemented and tested.

2. **Confidence-weighted overlap**
   - Same matching idea, but stronger-confidence detections count more.
   - Better for prioritizing likely-correct ingredients.
   - **Status:** implemented and tested.

3. **Penalty-aware (default)**
   - Starts with weighted overlap, then penalizes recipes needing many missing ingredients.
   - Balances “what matches now” vs “what user likely still needs.”
   - **Status:** implemented, tested, and used in app integration.

---

## Benchmark snapshot (latest)

### Retrieval-only benchmark (sample corpus sanity check)
- **Command run:** `python3 -m recipe_retrieval.cli eval --recipes fridge_data/sample_recipes.jsonl --cases fridge_data/eval_cases_sprint_b.jsonl --out runs/retrieval_eval`
- **Result summary:**
  - `overlap`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
  - `confidence_weighted`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
  - `penalty_aware`: recall@1=1.0, recall@3=1.0, recall@5=1.0, mrr=1.0
- **Interpretation:** retrieval plumbing remains stable on curated sample data.

### End-to-end benchmark on team eval set (new, more realistic)
- **Baseline run command:** `python3 eval_e2e.py --labels eval_data/labels.jsonl --out runs/e2e_eval`
  - **Baseline metrics:** `mean_f1=0.086`, `top1_rate=0.03`, `top3_rate=0.091`, `top5_rate=0.091`
  - **Artifact:** `runs/e2e_eval/e2e_eval_20260429T152239Z.json`
- **Controlled tuning run command:** `python3 eval_e2e.py --labels eval_data/labels.jsonl --out runs/e2e_eval --conf-threshold 0.15 --ranker overlap --top-k 5`
  - **Tuned metrics:** `mean_f1=0.108`, `top1_rate=0.061`, `top3_rate=0.091`, `top5_rate=0.091`
  - **Artifact:** `runs/e2e_eval/e2e_eval_20260429T152250Z.json`
- **Interpretation:** controlled tuning improved detection-side quality and top-1 retrieval hit rate, while top-3/top-5 remained unchanged.

### Failure-pattern summary (from E2E runs)
- 30/33 cases still miss at top-5 in both runs.
- Main bottleneck remains upstream detection quality/coverage (many low-F1 or zero-detection cases).
- Retrieval always returns candidates, but sample recipe corpus is too small/narrow for diverse `reasonable_recipes`.
- High-frequency detected labels in failed cases (`Tomato`, `Bacon`, `Bell Pepper`) indicate label-space mismatch with recipe coverage and alias vocabulary.

---

## Sprint status (A/B/C)

- **Sprint A (hardening):** completed.
  - Added edge-case tests, deterministic ordering, and clearer validation errors.
- **Sprint B (evaluation quality pass):** completed.
  - Expanded evaluation case set and reran ablations with artifact logging.
- **Sprint C (post-freeze tuning prep):** prepared.
  - Added config-driven tuning controls (`--tuning-config`, penalty flags, ranker selection).
  - Final tuning is blocked on dataset/vocabulary/detector freeze.

---

## Proposed feature (idea only, not implemented)

Optional user preference filters in retrieval:
- meal type (breakfast/lunch/dinner/snack),
- cuisine type,
- target time (or max-time bucket).

### Scope and dependency
- **In scope for my role:** yes (retrieval/ranking behavior).
- **Dependency:** requires reliable recipe metadata fields in the team dataset.
- **Current status:** proposed; no code changes for this feature yet.

---

## Team role deliverables status

### Group Members 1 & 3 — Ingredient Detection (Vision)
- **Deliverable:** image -> ingredient list with confidence.
  - **Status:** partially done (working baseline; quality still evolving).
- **Deliverable:** detection accuracy ownership and iteration.
  - **Status:** in progress.

### Group Members 1 & 3 — Data Processing and Normalization
- **Deliverable:** cleaned recipe dataset in stable query-ready schema/path.
  - **Status:** not finalized.
- **Deliverable:** shared ingredient vocabulary and alias map.
  - **Status:** not finalized.

### Group Member 2 (me) — Recipe Retrieval and Ranking
- **Deliverable:** ranked top-k retrieval from detected ingredient list.
  - **Status:** done (baseline integrated and verified).
- **Deliverable:** multiple rankers and retrieval-side eval/benchmarking.
  - **Status:** done for baseline/sample data; final pass pending dependency freeze.

### Group Member 4 — End-to-End Integration and Demo
- **Deliverable:** stitched pipeline and demo flow.
  - **Status:** partially done (semi-running path works).
- **Deliverable:** final demo polish and presentation flow.
  - **Status:** in progress.

---

## Ordered deliverables by owner

1. Ingredient detection output contract (ingredient + confidence) — Group Members 1 & 3  
2. Recipe dataset in agreed schema/path — Group Members 1 & 3  
3. Shared alias/vocabulary mapping — Group Members 1 & 3  
4. Retrieval/ranking engine (top-k + score breakdown) — Group Member 2 (me)  
5. Retrieval evaluation and benchmark artifacts — Group Member 2 (me)  
6. End-to-end integration wiring (upload -> detect -> rank -> display) — Group Member 4  
7. End-to-end validation + demo polish — Group Member 4  
8. Final retrieval tuning pass after dependency freeze — Group Member 2 (me), dependent on items 2-3 and detector stability  

---

## Ready-for-demo checklist

- [x] App uses real retrieval module (not retrieval stub)
- [x] Detections flow into ranking and show recipe suggestions
- [x] Retrieval unit tests pass
- [x] Evaluation CLI produces benchmark artifacts
- [ ] Final dataset path/schema finalized by data/normalization owners
- [ ] Shared alias map finalized by data/normalization owners
- [ ] Final detector output conventions frozen by vision owners

---

## Deliverables log (chronological evidence)

| Date (UTC) | What I delivered | Where |
|------------|------------------|-------|
| 2026-04-22 | Types for query, normalized tokens, ranked results, score breakdown | `recipe_retrieval/schema.py` |
| 2026-04-22 | Normalization layer + optional alias map support | `recipe_retrieval/normalize.py` |
| 2026-04-22 | Recipe loading (JSON/JSONL) and record model | `recipe_retrieval/corpus.py` |
| 2026-04-22 | Ingredient key normalization for matching | `recipe_retrieval/text.py` |
| 2026-04-22 | Inverted index for candidate generation | `recipe_retrieval/index.py` |
| 2026-04-22 | Three retrieval strategies (overlap, confidence-weighted, penalty-aware) | `recipe_retrieval/rankers.py` |
| 2026-04-22 | Core retrieval entrypoints (`build_index_from_paths`, `retrieve`) | `recipe_retrieval/pipeline.py` |
| 2026-04-22 | Retrieval eval/ablation framework and artifacts | `recipe_retrieval/eval.py` |
| 2026-04-22 | Retrieval CLI (`demo`, `eval`) | `recipe_retrieval/cli.py` |
| 2026-04-22 | Integration helper for post-freeze alias map | `recipe_retrieval/integrate.py` |
| 2026-04-22 | Package exports | `recipe_retrieval/__init__.py` |
| 2026-04-22 | Sample corpus and sample eval/query files | `fridge_data/sample_recipes.jsonl`, `fridge_data/eval_cases.jsonl`, `fridge_data/sample_query.json`, `fridge_data/sample_alias_map.json` |
| 2026-04-26 | Synced branch with latest `main` and reconciled `data/` -> `fridge_data/` paths | `README.md`, `tests/test_retrieval.py`, `recipe_retrieval/cli.py`, `recipe_retrieval/integrate.py`, `Martin-work-done.md` |
| 2026-04-26 | Wired Streamlit demo retrieval path to real ranking module | `app.py` |
| 2026-04-26 | Verified integration quality gate (tests + eval + app behavior) | `tests/test_retrieval.py`, `recipe_retrieval/cli.py`, `app.py`, `Martin-work-done.md` |
| 2026-04-26 | Sprint B eval pass with expanded case set | `fridge_data/eval_cases_sprint_b.jsonl`, `runs/retrieval_eval/retrieval_eval_20260426T190805Z.json`, `Martin-work-done.md` |
| 2026-04-26 | Sprint C prep with config-driven tuning surface | `recipe_retrieval/cli.py`, `fridge_data/retrieval_tuning_config.example.json`, `tests/test_retrieval.py`, `Martin-work-done.md` |

---

## Next 7-day action list

1. Get final dataset path/schema from Group Members 1 & 3 (required before final retrieval tuning).
2. Get finalized alias map from Group Members 1 & 3 (required to reduce label-vocabulary mismatch).
3. Confirm detector output conventions with vision owners (required for stable tuning baseline).
4. Run a broader tuning sweep in `eval_e2e.py` now that tuning flags are available.
5. Record before/after metrics and top recurring failure buckets for team handoff.
6. Update this doc with tuned defaults and final demo-ready benchmark notes.

---

## If I need something from the team right now

1. **From Group Members 1 & 3:** final recipe dataset schema/path and alias map draft.
   - **Reason:** retrieval quality ceiling is currently capped by dataset/label mismatch.
   - **What comes after:** I rerun tuning and benchmark against that finalized data.
2. **From vision owners:** confirmation of detector label/threshold conventions for final runs.
   - **Reason:** prevents tuning to a moving target.
   - **What comes after:** I lock retrieval defaults and publish final recommendation settings.

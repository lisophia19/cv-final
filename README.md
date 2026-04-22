# cv-final
Fridge food detection for recipe generation.

## Recipe retrieval and ranking (Martin)

`recipe_retrieval` takes detected ingredients (same shape as `test.py` after deduplication) and returns a ranked list of recipes. Use your own **JSON/JSONL** recipe corpus, or the bundled sample: `data/sample_recipes.jsonl`.

**Integration API**

```python
from recipe_retrieval import build_index_from_paths, retrieve

index = build_index_from_paths(["data/sample_recipes.jsonl"])
out = retrieve(
    [{"ingredient": "chicken", "confidence": 0.9}, {"ingredient": "rice", "confidence": 0.8}],
    index=index,
    ranker="penalty_aware",  # "overlap" | "confidence_weighted" | "penalty_aware"
    k=5,
)
# out.top_k: list of recipe_id, title, score, breakdown
```

**After the team freezes shared vocabulary** (optional alias map JSON: lowercase raw → canonical), use:

```python
from recipe_retrieval import retrieve_with_reconciled_vocab

# Commit data/team_ingredient_alias.json or pass alias_path=...
out = retrieve_with_reconciled_vocab(detections, index=index, k=5)
```

**CLI**

```bash
# Ablation (writes JSON under runs/retrieval_eval/)
python3 -m recipe_retrieval.cli eval \
  --recipes data/sample_recipes.jsonl \
  --cases data/eval_cases.jsonl \
  --out runs/retrieval_eval

# One-off demo
python3 -m recipe_retrieval.cli demo \
  --recipes data/sample_recipes.jsonl \
  --query data/sample_query.json \
  --ranker penalty_aware
```

**Unit tests** (from repo root):

```bash
python3 -m unittest discover -s tests -v
```

## Instructions to run code (vision / YOLO)
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -U ultralytics
   ```
3. Run the script:
   ```bash
   python yolo.py
   ```

## Notes
- If you get an import warning in the editor, select `.venv/bin/python` as your Python interpreter.
- Update `path/to/dataset.yaml` in `yolo.py` to your actual dataset config before training.
- For `test.py` (Roboflow), install `inference-sdk` and `python-dotenv` and set `ROBOFLOW_API_KEY` in a `.env` file.

# cv-final: What's in my Fridge?
Fridge food detection for recipe generation.

## Instructions to run code
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install -U pip
   python -m pip install -r requirements.txt
   ```
3. (Optional) Run notebooks 
4. Launch the full app:
   ```bash
   streamlit run app.py
   ```

## Notes
- If you get an import warning in the editor, select `.venv/bin/python` as your Python interpreter.
- Update `path/to/dataset.yaml` in `yolo.py` to your actual dataset config before training.
- For `test.py` (Roboflow), install `inference-sdk` and `python-dotenv` and set `ROBOFLOW_API_KEY` in a `.env` file.

## Portions 
### Datasets (Sophia and Bryant)
- `ingredients_data`: Roboflow ingredient dataset cleaned to 4.2k+ images across 113 classes.
- `eval_data`: Fridge/recipe evaluation set used for project metrics.

### Ingredient identification/classification - YOLO and DINO (Sophia)

- `yolo.py` and `yolo.ipynb`: train and experiment with YOLOv11 on the ingredient dataset, then export ingredient detections/crops. Outputs metrics that are later used to compare all three pipelines
- `dino.ipynb`: added metric calculations for YOLO/DINO pipeline
- `dino_eval_stat.ran.ipynb`: notebook used to inspect/evaluate full DINO pipeline

### Ingredient identification/classification - DINOv2 (Nina)

`dino/dino_test.py`: (file used to get familiar with using pretrained DINOv2 on sample images) loads a pretrained DINOv2 model and uses it to generate embeddings for all images in the data/ directory. It then computes pairwise cosine similarity across all images for ingredient classification.

`dino/dino.ipynb`: (actual DINO pipeline) contains the full ingredient classification pipeline:

- **Cell 1 — imports**

- **Cell 2 — Fully-frozen DINOv2 baseline**: loads the frozen pretrained DINOv2-base model, crops individual ingredients from labeled dataset images using YOLO bounding boxes, extracts a 768-dim CLS token embedding per crop, normalizes all embeddings, and saves them to a `.npz` file for k-NN classification. 85.3% accuracy.

- **Cell 3 — LoRA fine-tuning**: fine-tunes DINOv2-base with LoRA, applied to the query and value attention projections. Builds a `CropDataset` from YOLO-labeled images, trains a linear classification head on top of the LoRA-adapted backbone for 10 epochs using AdamW, then re-extracts and saves embeddings for k-NN evaluation. Achieved 88.7% accuracy with best LoRA parameters (rank=8, alpha=8, dropout=0.1).

- **Cell 4 — k-NN evaluation**: loads train and test `.npz` embedding files and runs k-NN classification (k=5, cosine similarity) on the test set. For each test crop, finds the 5 most similar training embeddings and takes a majority vote. Logs per-sample predictions and saves full results to `knn_results.txt`.

- **Cell 5 — Single image inference**: loads the saved LoRA adapters and classifier head and runs inference on a single image, printing the predicted class, confidence, and a probabilities for all 113 classes.

- **Cell 6 — Chunked embedding extraction**: memory-efficient version of Cell 2 for larger datasets. Extracts embeddings in batches and saves them to numbered chunk files (`chunk_0.npz`, `chunk_1.npz`, ...) every 1000 crops to avoid memory overflow.

- **Cell 7 — Merge chunks**: merges all chunk `.npz` files into a single `.npz` file for use in k-NN evaluation.

`dino/<folders>`: contains folders with the results of different training configurations tested during fine-tuning. Each folder contains `knn_results.txt` with the k-NN accuracy on the test set and `config.txt` with the LoRA parameters used for that run.

`dino/lora/`: contains the best performing LoRA configuration (rank=8, alpha=8, dropout=0.1, 88.7% accuracy), including the LoRA adapter weights (`dinov2_lora_adapters/`), the trained linear classifier head (`classifier/`), the train and test embeddings (`dinov2_objects_train.npz` and `dinov2_objects_test.npz`), and the corresponding `config.txt` and `knn_results.txt`.




### Recipe retrieval and ranking (Martin)

`recipe_retrieval` takes detected ingredients (same shape as `test.py` after deduplication) and returns a ranked list of recipes. Use your own **JSON/JSONL** recipe corpus, or the bundled sample: `fridge_data/sample_recipes.jsonl`.

**Integration API**

```python
from recipe_retrieval import build_index_from_paths, retrieve

index = build_index_from_paths(["fridge_data/sample_recipes.jsonl"])
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

# Commit fridge_data/team_ingredient_alias.json or pass alias_path=...
out = retrieve_with_reconciled_vocab(detections, index=index, k=5)
```

**CLI**

```bash
# Ablation (writes JSON under runs/retrieval_eval/)
python3 -m recipe_retrieval.cli eval \
  --recipes fridge_data/sample_recipes.jsonl \
  --cases fridge_data/eval_cases.jsonl \
  --out runs/retrieval_eval

# One-off demo
python3 -m recipe_retrieval.cli demo \
  --recipes fridge_data/sample_recipes.jsonl \
  --query fridge_data/sample_query.json \
  --ranker penalty_aware
```

**Unit tests** (from repo root):

```bash
python3 -m unittest discover -s tests -v
```

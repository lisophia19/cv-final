# End-to-end evaluation set

Hand-labeled kitchen/fridge photos for measuring end-to-end pipeline accuracy
(detection → retrieval → ranked recipes).

## Files
- `labels.jsonl` — one JSON object per line, schema below
- `images/` — additional images go here. The 5 seed entries reference `fridge_data/fridge_test*.jpg` directly so they are not duplicated.

## Schema
```json
{
  "image": "path/to/image.jpg",
  "true_ingredients": ["tomato", "onion", "egg"],
  "reasonable_recipes": ["omelette", "shakshuka", "frittata"]
}
```

- `image`: path relative to the `cv-final/` root
- `true_ingredients`: every ingredient visibly present in the photo (lowercase, common name)
- `reasonable_recipes`: free-text recipe names a person could plausibly make from the photo

## How to add more
1. Drop new images into `eval_data/images/`
2. Append a labeled line to `labels.jsonl`
3. Re-run the eval script (`eval_e2e.py`)

## Target size
Around 25 images total. We have 5 seed entries just as a start. We will add more
in the near future. 

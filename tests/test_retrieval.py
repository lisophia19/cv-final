import json
import unittest
from pathlib import Path

from recipe_retrieval.corpus import load_recipes_from_jsonl
from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.pipeline import build_index_from_paths, retrieve
from recipe_retrieval.eval import run_ablation, load_eval_cases

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "fridge_data" / "sample_recipes.jsonl"
CASES = ROOT / "fridge_data" / "eval_cases.jsonl"


class TestRetrieval(unittest.TestCase):
    def test_index_builds(self) -> None:
        r = load_recipes_from_jsonl(SAMPLE)
        self.assertEqual(len(r), 5)
        idx = RecipeIndex.build(r)
        self.assertEqual(len(idx), 5)

    def test_retrieve_penalty_aware_finds_r2(self) -> None:
        idx = build_index_from_paths([SAMPLE])
        res = retrieve(
            [
                {"ingredient": "chicken", "confidence": 0.9},
                {"ingredient": "rice", "confidence": 0.8},
            ],
            index=idx,
            ranker="penalty_aware",
            k=3,
        )
        top_ids = [x.recipe_id for x in res.top_k]
        self.assertIn("r2", top_ids)

    def test_ablation_runs(self) -> None:
        idx = build_index_from_paths([SAMPLE])
        cases = load_eval_cases(CASES)
        runs = run_ablation(idx, cases, k_max=5)
        self.assertEqual(len(runs), 3)
        for a in runs:
            self.assertIn("recall_at_1", a.metrics)
            self.assertTrue(0.0 <= a.metrics["mrr"] <= 1.0)

    def test_cli_eval_json_artifact(self) -> None:
        from recipe_retrieval.eval import write_artifact, run_ablation
        import tempfile
        idx = build_index_from_paths([SAMPLE])
        cases = load_eval_cases(CASES)
        ab = run_ablation(idx, cases)
        with tempfile.TemporaryDirectory() as td:
            p = write_artifact(td, ablations=ab, extra_meta={"t": 1})
            j = json.loads(Path(p).read_text(encoding="utf-8"))
            self.assertIn("ablations", j)
            self.assertEqual(len(j["ablations"]), 3)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from recipe_retrieval.corpus import load_recipes_from_jsonl
from recipe_retrieval.index import RecipeIndex
from recipe_retrieval.pipeline import build_index_from_paths, retrieve
from recipe_retrieval.eval import run_ablation, load_eval_cases
from recipe_retrieval.cli import _build_penalty_config, _load_tuning_config

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
        idx = build_index_from_paths([SAMPLE])
        cases = load_eval_cases(CASES)
        ab = run_ablation(idx, cases)
        with tempfile.TemporaryDirectory() as td:
            p = write_artifact(td, ablations=ab, extra_meta={"t": 1})
            j = json.loads(Path(p).read_text(encoding="utf-8"))
            self.assertIn("ablations", j)
            self.assertEqual(len(j["ablations"]), 3)

    def test_retrieve_rejects_non_positive_k(self) -> None:
        idx = build_index_from_paths([SAMPLE])
        with self.assertRaisesRegex(ValueError, "k must be a positive integer"):
            retrieve(
                [{"ingredient": "chicken", "confidence": 0.9}],
                index=idx,
                ranker="penalty_aware",
                k=0,
            )

    def test_retrieve_handles_duplicate_and_unknown_ingredients(self) -> None:
        idx = build_index_from_paths([SAMPLE])
        res = retrieve(
            [
                {"ingredient": "chicken", "confidence": 0.9},
                {"ingredient": "chicken", "confidence": 0.4},
                {"ingredient": "unknown_thing", "confidence": 0.8},
            ],
            index=idx,
            ranker="penalty_aware",
            k=5,
        )
        self.assertGreaterEqual(len(res.top_k), 1)
        # Normalizer should dedupe repeated ingredient canonicals.
        canonicals = [n.canonical.lower() for n in res.normalized]
        self.assertEqual(canonicals.count("chicken"), 1)

    def test_build_index_requires_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one recipe corpus path is required"):
            build_index_from_paths([])

    def test_load_recipes_reports_invalid_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.jsonl"
            bad.write_text('{"id":"r1","title":"x","ingredients":["a"]}\n{"id":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON in"):
                load_recipes_from_jsonl(bad)

    def test_build_index_reports_empty_recipe_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Recipe file is empty"):
                build_index_from_paths([empty])

    def test_load_tuning_config_parses_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "cfg.json"
            cfg.write_text('{"missing_penalty":0.2,"missing_cap":1.0}', encoding="utf-8")
            data = _load_tuning_config(str(cfg))
            self.assertEqual(data["missing_penalty"], 0.2)
            self.assertEqual(data["missing_cap"], 1.0)

    def test_load_tuning_config_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "cfg.json"
            cfg.write_text('["not","object"]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tuning config must be a JSON object"):
                _load_tuning_config(str(cfg))

    def test_build_penalty_config_from_args(self) -> None:
        class Args:
            missing_penalty = 0.2
            missing_cap = 0.9
            no_query_weight_norm = True

        cfg = _build_penalty_config(Args())
        self.assertEqual(cfg.missing_penalty, 0.2)
        self.assertEqual(cfg.missing_cap, 0.9)
        self.assertFalse(cfg.use_query_weight_sum_norm)


if __name__ == "__main__":
    unittest.main()

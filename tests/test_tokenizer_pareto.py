"""Dominance directions, ties, missing cells, interval order and receipt integrity."""

from pathlib import Path
import tempfile
import unittest

from benchmarks import tokenizer_pareto as p
from benchmarks import receipt_archive as a
from benchmarks import run_research_experiments as h
from tests.test_vocabulary_scaling import matrix


def point(identifier="a", **changes):
    return {
        "condition_id": identifier,
        "bytes_per_token": 4.0,
        "byte_fallback_percent": 10.0,
        "tokens": 100,
        "actual_vocab_size": 8192,
        "training_seconds": 10.0,
        **changes,
    }


def scaling_fixture():
    payload = matrix()
    payload["assignments"] = {"source": {"test_access": "forbidden_not_opened"}}
    for condition in payload["conditions"]:
        condition["artifact_directory"] = f"{condition['tokenizer']}-{condition['vocab_budget']}"
        for row in condition["records"]:
            row.update(byte_fallback_percent=10.0, documents=1)
    return payload


class TokenizerParetoTests(unittest.TestCase):
    def test_retained_receipt_and_all_scope_classifications_recompute(self):
        directory = Path(p.__file__).parent
        with (
            a.retained_bundle(directory / "pareto" / "issue93") as root,
            a.retained_bundle(directory / "scaling" / "issue92") as scaling_root,
        ):
            self._check_retained(root, scaling_root)

    def _check_retained(self, root, scaling_root):
        receipt = h.read_json(root / "manifest.json")
        self.assertEqual(receipt["status"], "complete")
        for path, digest in receipt["artifacts"].items():
            self.assertEqual(h.file_hash(root / path), digest, path)
        result = h.read_json(root / "results.json")
        source = scaling_root / "results.json"
        measurement, _ = p.verified_scaling(source)
        self.assertEqual(result["source_results_sha256"], h.file_hash(source))
        self.assertEqual(result["source_manifest_sha256"], h.file_hash(source.parent / "manifest.json"))
        self.assertEqual(result["measurement_identity"], measurement["identity"])
        self.assertEqual(result["classifications"], p.analyze(measurement))

    def test_all_five_directions_and_strictness(self):
        baseline = point("b")
        for field, value in (
            ("bytes_per_token", 4.1),
            ("byte_fallback_percent", 9.0),
            ("tokens", 99),
            ("actual_vocab_size", 8191),
            ("training_seconds", 9.0),
        ):
            with self.subTest(field=field):
                improved = point("a", **{field: value})
                self.assertTrue(p.dominates(improved, baseline))
                self.assertFalse(p.dominates(baseline, improved))
        self.assertFalse(p.dominates(baseline, baseline))

    def test_exact_ties_tradeoffs_and_dominator_lists(self):
        rows = p.classify(
            [
                point("a"),
                point("tie"),
                point("bad", bytes_per_token=3.0),
                point("tradeoff", bytes_per_token=5.0, training_seconds=20.0),
            ]
        )
        lookup = {r["condition_id"]: r for r in rows}
        self.assertEqual(lookup["a"]["exact_ties"], ["tie"])
        self.assertEqual(lookup["bad"]["dominators"], ["a", "tie"])
        self.assertEqual(lookup["bad"]["classification"], "dominated")
        self.assertEqual(lookup["tradeoff"]["classification"], "non_dominated")

    def test_uncertainty_is_interval_dominance_not_epsilon(self):
        a, b = point("a", training_seconds=9.9), point("b")
        self.assertTrue(p.dominates(a, b))
        self.assertFalse(p.dominates(a, b, 0.01))
        self.assertTrue(p.dominates(point(training_seconds=9.0), b, 0.05))
        points = [point(str(i), training_seconds=9 + i * 0.2) for i in range(8)]
        for fraction in p.SENSITIVITY:
            for left in points:
                for right in points:
                    if p.dominates(left, right, fraction):
                        self.assertFalse(p.dominates(right, left, fraction))
                        for third in points:
                            if p.dominates(right, third, fraction):
                                self.assertTrue(p.dominates(left, third, fraction))

    def test_invalid_axes_and_duplicate_ids_rejected(self):
        for changes in (
            {"tokens": None},
            {"bytes_per_token": float("nan")},
            {"training_seconds": float("inf")},
            {"actual_vocab_size": 0},
            {"byte_fallback_percent": 101},
            {"tokens": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                p.classify([point(**changes)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.classify([point(), point()])

    def test_complete_and_missing_cells_remain_scope_specific(self):
        payload = scaling_fixture()
        failed = payload["conditions"][0]
        failed.update(status="budget_not_reached", records=[])
        rows = p.analyze(payload)
        scopes = {(r["scope"], r["domain"], r["language"]) for r in rows}
        self.assertEqual(len(rows), len(scopes) * 15 * 3)
        for key in scopes:
            for fraction in p.SENSITIVITY:
                selected = [
                    r
                    for r in rows
                    if (r["scope"], r["domain"], r["language"]) == key and r["timing_sensitivity_fraction"] == fraction
                ]
                self.assertEqual(len(selected), 15)
                missing = [r for r in selected if r["status"] != "complete"]
                self.assertEqual(len(missing), 1)
                self.assertEqual(missing[0]["classification"], "not_classified_missing_condition")
                self.assertNotIn("bytes_per_token", missing[0])

    def test_different_source_denominators_rejected(self):
        payload = scaling_fixture()
        payload["conditions"][0]["records"][0]["documents"] = 2
        # The first fixture record is validation or train depending on fixture ordering.
        for row in payload["conditions"][0]["records"]:
            if row["split"] == "validation":
                row["documents"] = 2
        with self.assertRaisesRegex(ValueError, "different validation inputs"):
            p.analyze(payload)

    def test_receipt_tampering_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results.json"
            h.write_new_json(path, scaling_fixture())
            h.write_new_json(root / "manifest.json", {"status": "complete", "artifacts": {"results.json": "0" * 64}})
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                p.verified_scaling(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "results.json"
            h.write_new_json(path, scaling_fixture())
            h.write_new_json(
                root / "manifest.json",
                {"status": "complete", "artifacts": {"results.json": h.file_hash(path), "../outside": "0" * 64}},
            )
            with self.assertRaisesRegex(ValueError, "escape"):
                p.verified_scaling(path)

    def test_zero_success_is_explicitly_unavailable(self):
        payload = scaling_fixture()
        for condition in payload["conditions"]:
            condition.update(status="resource_limit", records=[])
        with self.assertRaisesRegex(ValueError, "no complete"):
            p.analyze(payload)


if __name__ == "__main__":
    unittest.main()

"""Tests for Python-Rust boundary overhead profiling, parity gates, and optimizations (Issue #101)."""

import hashlib
import json
from pathlib import Path
import unittest

from benchmarks.profile_boundary_overhead import (
    FIXTURES,
    baseline_requires_python_security,
    decompose_workload_stages,
    make_benchmark_tokenizer,
    optimized_requires_python_security,
    run_crossing_1_string_utf8_access,
    run_crossing_2_token_ids_to_containers,
    run_crossing_3_batch_output_construction,
    run_crossing_4_python_object_creation,
    run_crossing_5_repeated_ffi_crossings,
    run_exact_parity_gate,
)
from uniqtoken._native import native_text_supported


class BoundaryOverheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_benchmark_tokenizer()

    def test_native_text_supported_ascii_and_surrogates(self):
        self.assertTrue(native_text_supported("hello world"))
        self.assertTrue(native_text_supported("12345!@#$%^&*()"))
        self.assertTrue(native_text_supported("Café \u4e2d\u6587"))
        self.assertFalse(native_text_supported("corrupted \ud800 surrogate"))

    def test_security_check_parity_across_text_variants(self):
        test_cases = [
            "Normal plain ASCII text",
            "Café \u4e2d\u6587 text without markers",
            "Text with <|special_control_token|>",
            "Text with \uff1c|fullwidth_angle_bracket|>",
            "Text with < but no pipe",
            "Text with | but no angle bracket",
            "Text with \ue000 metaspace escape",
            "Text with \ue001 metaspace escape",
            "Text with \ud800 lone surrogate",
        ]
        for s in test_cases:
            base = baseline_requires_python_security(s)
            opt = optimized_requires_python_security(s)
            self.assertEqual(base, opt, f"Security parity mismatch on: {s!r}")

    def test_exact_parity_gate_passes_on_fixtures(self):
        passed = run_exact_parity_gate(self.tokenizer, FIXTURES)
        self.assertTrue(passed)

    def test_stage_decomposition_covers_all_components(self):
        dec = decompose_workload_stages(self.tokenizer, "test_decomp", [FIXTURES["short"]])
        self.assertGreater(dec.total_latency_ms, 0.0)
        self.assertGreater(dec.input_conversion_ms, 0.0)
        self.assertGreater(dec.native_compute_ms, 0.0)
        self.assertGreater(dec.materialization_ms, 0.0)
        total_pct = dec.input_conversion_pct + dec.native_compute_pct + dec.materialization_pct
        self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

    def test_microbenchmarks_five_crossings(self):
        c1 = run_crossing_1_string_utf8_access(warmup=1, reps=2)
        self.assertIn("baseline", c1)
        self.assertIn("optimized", c1)
        self.assertEqual(c1["optimized"].copied_bytes, 0)

        c2 = run_crossing_2_token_ids_to_containers(warmup=1, reps=2)
        self.assertIn("ids_64", c2)
        self.assertIn("ids_512", c2)

        c3 = run_crossing_3_batch_output_construction(warmup=1, reps=2)
        self.assertIn("batch_1", c3)
        self.assertIn("batch_32", c3)

        c4 = run_crossing_4_python_object_creation(warmup=1, reps=2)
        self.assertIn("ints", c4)
        self.assertIn("strings", c4)

        c5 = run_crossing_5_repeated_ffi_crossings(self.tokenizer, warmup=1, reps=2)
        self.assertIn("baseline_precheck", c5)
        self.assertIn("optimized_precheck", c5)
        self.assertIn("iterative_crossing", c5)
        self.assertIn("batched_crossing", c5)

    def test_manifest_integrity_and_receipt_hashes(self):
        out_dir = Path("benchmarks/boundary_overhead/issue101")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json must exist")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["issue"], 101)

        for filename, expected_hash in manifest["published_files"].items():
            target_path = out_dir / filename
            self.assertTrue(target_path.exists(), f"{filename} must exist")
            actual_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, f"Hash mismatch for {filename}")


if __name__ == "__main__":
    unittest.main()

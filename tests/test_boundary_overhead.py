"""Guard equivalence, benchmark validity, and published diagnostic integrity."""

from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unicodedata
import unittest
from unittest.mock import patch

from benchmarks import profile_boundary_overhead as profiler
from uniqtoken import _native, pre_tokenizer, seed_builder, unigram_trainer
from uniqtoken.tokenizer import CustomTokenizer
import uniqtoken.tokenizer as tokenizer_module


class BoundaryOverheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = profiler.make_benchmark_tokenizer()

    def test_native_support_matches_baseline(self):
        for text in ("", "plain ASCII", "Cafe\u0301 \u4e2d\u6587", "\ud800", "abc\udfff"):
            with self.subTest(text=repr(text)):
                self.assertEqual(_native.native_text_supported(text), profiler.baseline_native_text_supported(text))

    def test_nfkc_delimiter_sources_exhaustively(self):
        # Exercise the active Unicode database on every Python version in CI.
        sources = {"<": set(), "|": set()}
        for codepoint in range(sys.maxunicode + 1):
            char = chr(codepoint)
            normalized = unicodedata.normalize("NFKC", char)
            for delimiter in sources:
                if delimiter in normalized:
                    sources[delimiter].add(char)
        self.assertEqual(sources["<"], {"<", "\ufe64", "\uff1c"})
        self.assertEqual(sources["|"], {"|", "\uff5c"})
        for left, right, middle in itertools.product(sources["<"], sources["|"], ("", "\u0301", "\u200d", " ")):
            text = f"prefix {left}{middle}{right}unk|> suffix"
            self.assertEqual(
                CustomTokenizer._requires_python_security(text), profiler.baseline_requires_python_security(text)
            )

    def test_security_guard_equivalence(self):
        for text in (
            "ASCII < without pipe",
            "ASCII | without angle",
            "<|unk|>",
            "\ufe64\uff5cunk\uff5c\uff1e",
            "\uff1c|unknown|>",
            "\ue000",
            "\ue001",
            "\ud800",
            "emoji \U0001f469\u200d\U0001f4bb",
        ):
            self.assertEqual(
                CustomTokenizer._requires_python_security(text), profiler.baseline_requires_python_security(text)
            )

    def test_baseline_context_restores_every_imported_helper_on_error(self):
        modules = (_native, tokenizer_module, pre_tokenizer, seed_builder, unigram_trainer)
        originals = [module.native_text_supported for module in modules]
        security = CustomTokenizer._requires_python_security
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with profiler.implementation("baseline"):
                for module in modules:
                    self.assertIs(module.native_text_supported, profiler.baseline_native_text_supported)
                self.assertIs(CustomTokenizer._requires_python_security, profiler.baseline_requires_python_security)
                raise RuntimeError("injected")
        self.assertEqual([module.native_text_supported for module in modules], originals)
        self.assertIs(CustomTokenizer._requires_python_security, security)

    def test_exact_parity_gate_and_security_errors(self):
        records = profiler.run_exact_parity_gate(self.tokenizer)
        self.assertEqual(len(records), len(profiler.FIXTURES) + 24)
        for variant in ("baseline", "optimized"):
            with profiler.implementation(variant):
                snapshot = profiler.snapshot(self.tokenizer, "<|unk|>", {"disallowed_special_action": "raise"})
                self.assertEqual(snapshot["ids"]["error"], "ValueError")

    def test_parity_gate_rejects_decode_drift(self):
        original = self.tokenizer.decode

        def drift(ids):
            result = original(ids)
            return (
                result
                if CustomTokenizer._requires_python_security is profiler.baseline_requires_python_security
                else result + "!"
            )

        with patch.object(self.tokenizer, "decode", side_effect=drift):
            with self.assertRaisesRegex(AssertionError, "API parity failed"):
                profiler.run_exact_parity_gate(self.tokenizer)

    def test_native_probes_fail_loudly_when_native_unavailable(self):
        with patch.object(tokenizer_module, "_native_core", None):
            with self.assertRaisesRegex(RuntimeError, "native IDs APIs"):
                profiler.run_crossing_5_repeated_ffi_crossings(self.tokenizer, warmup=0, reps=1)

    def test_native_call_counts_and_batch_output(self):
        if tokenizer_module._native_core is None:
            self.skipTest("native extension not installed")
        records = profiler.run_crossing_5_repeated_ffi_crossings(self.tokenizer, warmup=0, reps=2)
        self.assertEqual(records["iterative"]["native_calls"], {profiler.NATIVE_IDS[0]: 32})
        self.assertEqual(records["batch"]["native_calls"], {profiler.NATIVE_IDS[1]: 1})

    def test_macro_times_real_api_with_checks_inside_only(self):
        if tokenizer_module._native_core is None:
            self.skipTest("native extension not installed")
        seen = []
        actual = profiler.implementation

        @contextmanager
        def spy(variant):
            with actual(variant):
                helper = CustomTokenizer._requires_python_security

                def count(text):
                    seen.append(variant)
                    return helper(text)

                with patch.object(CustomTokenizer, "_requires_python_security", staticmethod(count)):
                    yield

        with patch.object(profiler, "implementation", spy):
            rows = profiler.run_macrobenchmark_matrix(
                self.tokenizer, warmup=0, reps=2, cases=[("short_single", [profiler.FIXTURES["short"]])]
            )
        # One path check + one memory call + 2 trials of 5 actual calls, per variant.
        self.assertEqual(seen.count("baseline"), 12)
        self.assertEqual(seen.count("optimized"), 12)
        self.assertEqual(rows[0]["native_calls"]["baseline"], {profiler.NATIVE_IDS[0]: 1})

    def test_diagnostics_are_independent_and_native_stages_unknown(self):
        result = profiler.measure_boundary_diagnostics(self.tokenizer, [profiler.FIXTURES["short"]], warmup=0, reps=2)
        self.assertIsNone(result["native_compute_ms"])
        self.assertIsNone(result["native_materialization_ms"])
        self.assertIn("not additive", result["scope"])
        self.assertGreater(result["public_ids_batch"]["p50_ms"], 0)

    def test_python_probes_do_not_claim_native_measurements(self):
        for function in (
            profiler.run_crossing_1_string_utf8_access,
            profiler.run_crossing_2_token_ids_to_containers,
            profiler.run_crossing_3_batch_output_construction,
            profiler.run_crossing_4_python_object_creation,
        ):
            for record in function(warmup=0, reps=2).values():
                self.assertEqual(record["native_calls"], 0)
                self.assertIsNone(record["copied_native_bytes"])
                self.assertIn("Python-only", record["scope"])
                self.assertGreaterEqual(record["memory"]["peak_bytes"], record["memory"]["retained_bytes"])

    def test_memory_probe_releases_tracing_on_error(self):
        def fail():
            raise RuntimeError("injected")

        with self.assertRaisesRegex(RuntimeError, "injected"):
            profiler.python_memory(fail)
        self.assertFalse(tracemalloc.is_tracing())

    def test_publish_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                profiler.publish({}, Path(directory))

    def test_published_manifest_and_report_match_results(self):
        directory = profiler.ROOT / "benchmarks/boundary_overhead/issue101"
        manifest = json.loads((directory / "manifest.json").read_bytes())
        self.assertEqual(manifest["issue"], 101)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(set(manifest["published_files"]), {"results.json", "REPORT.md"})
        for filename, expected in manifest["published_files"].items():
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), expected)
        payload = json.loads((directory / "results.json").read_bytes())
        self.assertEqual(payload["issue_completion"], "partial")
        self.assertEqual(len(payload["macrobenchmarks"]), 10)
        self.assertEqual(payload["metadata"]["fixture_sha256"], profiler.digest(profiler.FIXTURES))
        for filename, expected in payload["metadata"]["source_sha256"].items():
            self.assertEqual(profiler.sha256(profiler.ROOT / filename), expected)
        self.assertEqual((directory / "REPORT.md").read_bytes(), profiler.generate_markdown_report(payload).encode())


if __name__ == "__main__":
    unittest.main()

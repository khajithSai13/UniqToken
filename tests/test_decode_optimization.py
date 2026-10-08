"""Unit tests for decode and ID-to-text reconstruction optimizations (Issue #103)."""

import hashlib
import json
from pathlib import Path
import unittest

from benchmarks.profile_decode_optimization import (
    FIXTURES,
    make_benchmark_tokenizer,
    profile_dimension_1_token_id_lookup,
    profile_dimension_2_special_tokens,
    profile_dimension_3_byte_fallback,
    profile_dimension_4_piece_metaspace,
    profile_dimension_5_string_concatenation,
    profile_dimension_6_batch_scheduling,
    profile_dimension_7_streaming_vs_non_streaming,
    run_exact_parity_gate,
)
from uniqtoken.byte_codec import BYTE_TOKEN_TO_VAL, ByteFallbackEngine


class DecodeOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_benchmark_tokenizer()

    def test_exact_parity_gate_passes(self):
        passed = run_exact_parity_gate(self.tokenizer)
        self.assertTrue(passed)

    def test_roundtrip_all_fixtures(self):
        for name, text in FIXTURES.items():
            if name == "special_tokens":
                continue
            ids = self.tokenizer.encode_to_ids(text)
            decoded = self.tokenizer.decode(ids)
            expected = self.tokenizer.normalizer.restore_escaped_metaspace(
                self.tokenizer.normalizer.normalize(text).replace(self.tokenizer.normalizer.space_char, " ")
            )
            self.assertEqual(decoded, expected, f"Roundtrip failed for {name}")

    def test_byte_fallback_table_and_error_handling(self):
        # 256 byte values covered with hex variants
        self.assertEqual(len(BYTE_TOKEN_TO_VAL), 412)
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0x00>"))
        self.assertTrue(ByteFallbackEngine.is_byte_token("<0xFF>"))
        self.assertFalse(ByteFallbackEngine.is_byte_token("<0xGG>"))
        self.assertFalse(ByteFallbackEngine.is_byte_token("hello"))

        # Valid UTF-8 byte tokens
        valid_bytes = ["<0x41>", "<0x42>", "<0x43>"]  # 'ABC'
        self.assertEqual(ByteFallbackEngine.decode_tokens(valid_bytes), "ABC")

        # Invalid UTF-8 sequence must raise UnicodeDecodeError
        invalid_bytes = ["<0xFF>", "<0xFF>"]
        with self.assertRaises(UnicodeDecodeError):
            ByteFallbackEngine.decode_tokens(invalid_bytes)

    def test_strict_mode_raises_on_unknown_id(self):
        # Lenient: does not raise
        lenient = self.tokenizer.decode([999999], strict=False)
        self.assertIsInstance(lenient, str)

        # Strict: raises ValueError
        with self.assertRaises(ValueError):
            self.tokenizer.decode([999999], strict=True)

    def test_metaspace_fast_path(self):
        normalizer = self.tokenizer.normalizer
        clean_text = "Simple clean text without any escapes"
        # Bypasses loop entirely and returns same object or string
        self.assertEqual(normalizer.restore_escaped_metaspace(clean_text), clean_text)

        # Escaped text restores accurately
        escaped_text = f"prefix{normalizer._ESCAPE_PREFIX}{normalizer._ESCAPED_METASPACE}suffix"
        restored = normalizer.restore_escaped_metaspace(escaped_text)
        self.assertEqual(restored, f"prefix{normalizer.space_char}suffix")

    def test_streaming_decoder_parity(self):
        text = "Streaming test with emojis 🚀🎉 and words!"
        ids = self.tokenizer.encode_to_ids(text)
        sd = self.tokenizer.get_streaming_decoder()
        deltas = [sd.feed_token_id(t) for t in ids]
        deltas.append(sd.flush())
        streamed = "".join(deltas)
        non_streamed = self.tokenizer.decode(ids)
        self.assertEqual(streamed, non_streamed)

    def test_seven_profiling_dimensions_execute(self):
        d1 = profile_dimension_1_token_id_lookup(self.tokenizer, warmup=1, reps=2)
        self.assertIn("lenient_lookup", d1)
        self.assertIn("strict_lookup", d1)

        d2 = profile_dimension_2_special_tokens(self.tokenizer, warmup=1, reps=2)
        self.assertIn("special_token_handling", d2)

        d3 = profile_dimension_3_byte_fallback(warmup=1, reps=2)
        self.assertIn("baseline_regex", d3)
        self.assertIn("optimized_table", d3)

        d4 = profile_dimension_4_piece_metaspace(self.tokenizer, warmup=1, reps=2)
        self.assertIn("baseline_char_loop", d4)
        self.assertIn("optimized_fast_path", d4)

        d5 = profile_dimension_5_string_concatenation(warmup=1, reps=2)
        self.assertIn("join_1000_segments", d5)

        d6 = profile_dimension_6_batch_scheduling(self.tokenizer, warmup=1, reps=2)
        self.assertIn("batch_1", d6)
        self.assertIn("batch_32", d6)

        d7 = profile_dimension_7_streaming_vs_non_streaming(self.tokenizer, warmup=1, reps=2)
        self.assertIn("non_streaming", d7)
        self.assertIn("streaming", d7)

    def test_manifest_receipt_hashes(self):
        out_dir = Path("benchmarks/decode_optimization/issue103")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json must exist")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["issue"], 103)

        for filename, expected_hash in manifest["published_files"].items():
            target_path = out_dir / filename
            self.assertTrue(target_path.exists(), f"{filename} must exist")
            actual_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, f"Hash mismatch for {filename}")


if __name__ == "__main__":
    unittest.main()

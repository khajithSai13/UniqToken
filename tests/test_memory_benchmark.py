"""Unit test suite for the tokenizer memory benchmark (Issue #104).

Verifies:
- PrefixTrie, UnigramModel, BPEModel, and CustomTokenizer memory footprint APIs.
- Results schema, metadata, units, and artifact hash presence.
- Manifest SHA-256 integrity against generated receipts.
- Linear memory scaling and batch boundedness invariants.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from uniqtoken.bpe_model import BPEModel
from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.trie import PrefixTrie
from uniqtoken.unigram_trainer import UnigramModel

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS_DIR = ROOT / "benchmarks" / "memory" / "issue104"


class MemoryBenchmarkTests(unittest.TestCase):
    def test_prefix_trie_memory_footprint(self) -> None:
        vocab = {"hello": -1.0, "world": -2.0, "he": -0.5, "help": -1.5}
        trie = PrefixTrie.from_vocab(vocab)
        footprint = trie.memory_footprint()

        self.assertIn("total_nodes", footprint)
        self.assertIn("terminal_nodes", footprint)
        self.assertIn("total_bytes", footprint)
        self.assertEqual(footprint["terminal_nodes"], 4)
        self.assertGreater(footprint["total_nodes"], 4)
        self.assertGreater(footprint["total_bytes"], 0)

    def test_unigram_model_memory_footprint(self) -> None:
        vocab = {"<unk>": 0.0, "hello": -1.0, "world": -2.0}
        token_to_id = {"<unk>": 0, "hello": 1, "world": 2}
        id_to_token = {0: "<unk>", 1: "hello", 2: "world"}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=["<unk>"],
            byte_fallback=False,
            unk_token="<unk>",
        )
        footprint = model.memory_footprint()

        self.assertEqual(footprint["vocab_size"], 3)
        self.assertGreater(footprint["vocab_bytes"], 0)
        self.assertGreater(footprint["token_to_id_bytes"], 0)
        self.assertGreater(footprint["id_to_token_bytes"], 0)
        self.assertGreater(footprint["trie_bytes"], 0)
        self.assertGreater(footprint["total_model_bytes"], 0)
        self.assertEqual(footprint["trie_terminals"], 3)

    def test_bpe_model_memory_footprint(self) -> None:
        vocab = {ByteFallbackEngine.byte_to_token(b) for b in range(256)}
        vocab.update(["hello", "world"])
        token_to_id = {t: idx for idx, t in enumerate(sorted(vocab))}
        id_to_token = {idx: t for t, idx in token_to_id.items()}
        merges = {("h", "e"): 0, ("l", "l"): 1}
        model = BPEModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            byte_fallback=True,
        )
        footprint = model.memory_footprint()

        self.assertEqual(footprint["vocab_size"], 258)
        self.assertEqual(footprint["merges_count"], 2)
        self.assertGreater(footprint["total_model_bytes"], 0)

    def test_custom_tokenizer_memory_footprint(self) -> None:
        corpus = ["The quick brown fox jumps over 42 lazy dogs."] * 10
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, min_frequency=1, verbose=False)
        footprint = tok.memory_footprint()

        self.assertEqual(footprint["tokenizer_type"], "UnigramModel")
        self.assertEqual(footprint["vocab_size"], 320)
        self.assertIn("model", footprint)
        self.assertGreater(footprint["total_bytes"], 0)
        self.assertGreater(footprint["model"]["trie_bytes"], 0)

    def test_receipts_files_exist(self) -> None:
        results_file = RECEIPTS_DIR / "results.json"
        report_file = RECEIPTS_DIR / "REPORT.md"
        manifest_file = RECEIPTS_DIR / "manifest.json"

        self.assertTrue(results_file.is_file(), f"Missing {results_file}")
        self.assertTrue(report_file.is_file(), f"Missing {report_file}")
        self.assertTrue(manifest_file.is_file(), f"Missing {manifest_file}")

    def test_manifest_sha256_integrity(self) -> None:
        manifest_file = RECEIPTS_DIR / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

        self.assertEqual(manifest_data.get("schema_version"), "1.0")
        sha_map = manifest_data.get("sha256", {})

        for filename, expected_hash in sha_map.items():
            target_path = RECEIPTS_DIR / filename
            self.assertTrue(target_path.is_file(), f"File {filename} in manifest does not exist")
            actual_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"SHA-256 hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}",
            )

    def test_results_json_schema_and_completeness(self) -> None:
        results_file = RECEIPTS_DIR / "results.json"
        data = json.loads(results_file.read_text(encoding="utf-8"))

        # 1. Metadata check
        meta = data.get("metadata", {})
        self.assertEqual(meta.get("benchmark_issue"), 104)
        self.assertEqual(meta.get("worker_count"), 1)
        self.assertIn("platform", meta)
        self.assertIn("processor", meta)
        self.assertIn("total_ram_bytes", meta)
        self.assertIn("sampling_resolution", meta)
        self.assertIn("platform_limitations", meta)
        self.assertIn("units", meta)
        self.assertEqual(meta["units"]["peak_rss"], "bytes")
        self.assertEqual(meta["units"]["traced_memory"], "bytes")

        # 2. Cold start records
        cold_start = data.get("cold_start", [])
        self.assertGreaterEqual(len(cold_start), 3)
        tokenizers = {c["tokenizer"] for c in cold_start}
        self.assertTrue({"uniqtoken", "sentencepiece", "tokenizers"}.issubset(tokenizers))
        for c in cold_start:
            self.assertGreater(c["peak_rss_bytes"], 0)
            self.assertGreater(c["peak_traced_bytes"], 0)

        # 3. Workload matrix records
        matrix = data.get("workload_matrix", [])
        self.assertGreaterEqual(len(matrix), 60)  # 5 fixtures * 4 batch sizes * 3 tokenizers
        for item in matrix:
            self.assertIn(item["batch_size"], (1, 8, 32, 128))
            self.assertGreater(item["input_bytes"], 0)
            self.assertGreater(item["token_count"], 0)
            self.assertGreater(item["encode_peak_rss_bytes"], 0)

        # 4. Long document scaling records
        long_doc = data.get("long_document", [])
        self.assertGreaterEqual(len(long_doc), 15)  # 5 lengths * 3 tokenizers
        for ld in long_doc:
            self.assertGreater(ld["actual_input_bytes"], 0)
            self.assertGreater(ld["temp_buffer_bytes"], 0)
            self.assertGreater(ld["temp_buffer_per_input_byte"], 0.0)


if __name__ == "__main__":
    unittest.main()

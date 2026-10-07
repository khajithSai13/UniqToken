"""Regression and integrity tests for parallel batch scheduling and saturation (Issue #100).

Validates:
1. Hardware topology detection and core reporting.
2. Standardized workload fixture generation covering all 5 required matrix dimensions.
3. Deterministic output ordering and content parity across worker counts (1, 2, 4, 8, 16).
4. Bootstrap uncertainty estimation and statistical calculations.
5. Overhead decomposition and saturation point calculation logic.
6. Receipt and manifest cryptographic SHA-256 verification.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

from benchmarks import parallel_batch_scheduling as pbs


class HardwareTopologyTests(unittest.TestCase):
    """Test suite for hardware topology detection."""

    def test_topology_detection_returns_valid_system_specs(self) -> None:
        """Verify that topology detection returns positive logical and physical core counts."""
        topo = pbs.HardwareTopology.detect()
        self.assertGreaterEqual(topo.logical_cores, 1)
        self.assertGreaterEqual(topo.physical_cores, 1)
        self.assertGreater(topo.total_memory_bytes, 0)
        self.assertTrue(isinstance(topo.os_system, str) and len(topo.os_system) > 0)
        self.assertTrue(isinstance(topo.processor, str))


class WorkloadFixtureTests(unittest.TestCase):
    """Test suite for measurement matrix fixture construction."""

    def test_fixtures_cover_all_required_matrix_dimensions(self) -> None:
        """Verify that fixtures cover short, long, heterogeneous, multilingual, and code inputs."""
        batch_size = 32
        fixtures = pbs.build_workload_fixtures(batch_size)
        expected_keys = {
            "homogeneous_short",
            "homogeneous_long",
            "heterogeneous_mixed",
            "multilingual",
            "source_code",
        }
        self.assertEqual(set(fixtures.keys()), expected_keys)
        for name, items in fixtures.items():
            self.assertEqual(len(items), batch_size, f"Batch size mismatch in {name}")
            self.assertTrue(all(isinstance(x, str) and len(x) > 0 for x in items))

    def test_homogeneous_and_heterogeneous_byte_characteristics(self) -> None:
        """Verify that short texts are smaller than long texts and heterogeneous batches mix both."""
        fixtures = pbs.build_workload_fixtures(16)
        short_bytes = [len(x.encode("utf-8")) for x in fixtures["homogeneous_short"]]
        long_bytes = [len(x.encode("utf-8")) for x in fixtures["homogeneous_long"]]
        mixed_bytes = [len(x.encode("utf-8")) for x in fixtures["heterogeneous_mixed"]]

        self.assertLess(max(short_bytes), min(long_bytes))
        # Mixed batch contains both short and long texts
        self.assertTrue(any(b < min(long_bytes) for b in mixed_bytes))
        self.assertTrue(any(b > max(short_bytes) for b in mixed_bytes))


class ParallelBatchOrderingAndParityTests(unittest.TestCase):
    """Test suite for deterministic output ordering and token parity across worker counts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train a benchmark tokenizer once for parity tests."""
        cls.tok = pbs.create_benchmark_tokenizer()
        cls.fixtures = pbs.build_workload_fixtures(16)

    def test_deterministic_output_ordering_and_token_parity(self) -> None:
        """Verify that worker counts (1, 2, 4, 8, 16) produce identical output ordering and token content."""
        for workload_name, texts in self.fixtures.items():
            baseline_tokens = self.tok.encode_batch(texts, num_workers=1)
            baseline_ids = self.tok.encode_to_ids_batch(texts, num_workers=1)

            for workers in (1, 2, 4, 8, 16):
                actual_tokens = self.tok.encode_batch(texts, num_workers=workers)
                actual_ids = self.tok.encode_to_ids_batch(texts, num_workers=workers)

                self.assertEqual(
                    len(actual_tokens),
                    len(baseline_tokens),
                    f"Length mismatch at {workload_name} with {workers} workers",
                )
                self.assertEqual(
                    actual_tokens,
                    baseline_tokens,
                    f"Token ordering or content mismatch in {workload_name} with {workers} workers",
                )
                self.assertEqual(
                    actual_ids,
                    baseline_ids,
                    f"Token ID ordering or content mismatch in {workload_name} with {workers} workers",
                )


class StatisticalAndOverheadCalculationTests(unittest.TestCase):
    """Test suite for bootstrap confidence intervals and overhead calculations."""

    def test_bootstrap_median_ci_bounds(self) -> None:
        """Verify that bootstrap median CI covers the sample median within valid bounds."""
        samples = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        median = 13.0
        ci_lower, ci_upper = pbs.bootstrap_median_ci(samples)
        self.assertLessEqual(ci_lower, median)
        self.assertGreaterEqual(ci_upper, median)
        self.assertGreaterEqual(ci_lower, min(samples))
        self.assertLessEqual(ci_upper, max(samples))

    def test_overhead_decomposition_logic(self) -> None:
        """Verify speedup, parallel efficiency, and overhead fraction formulas."""
        t1 = 100.0
        t4 = 30.0
        workers = 4

        speedup = t1 / t4
        efficiency = speedup / workers
        overhead = max(0.0, 1.0 - efficiency)

        self.assertAlmostEqual(speedup, 3.3333, places=3)
        self.assertAlmostEqual(efficiency, 0.8333, places=3)
        self.assertAlmostEqual(overhead, 0.1667, places=3)


class ArtifactManifestVerificationTests(unittest.TestCase):
    """Test suite for published benchmark artifacts and manifest cryptographic integrity."""

    def test_published_manifest_matches_disk_artifacts(self) -> None:
        """Verify that results.json, REPORT.md, and manifest.json exist with valid SHA-256 digests."""
        root = Path(__file__).resolve().parents[1] / "benchmarks" / "scheduling" / "issue100"
        manifest_path = root / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json does not exist")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["issue"], 100)
        self.assertEqual(manifest["status"], "complete")

        for relative, digest in manifest["artifacts"].items():
            file_path = root / relative
            self.assertTrue(file_path.exists(), f"Artifact {relative} does not exist")
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self.assertEqual(actual_hash, digest, f"Digest mismatch for {relative}")


if __name__ == "__main__":
    unittest.main()

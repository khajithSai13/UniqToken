"""Regression tests for UniqToken v2 candidate objective and protocol (Issue #94).

Validates:
1. Protocol schema, metadata, and research-integrity contracts in
   benchmarks/protocols/superbpe_v2_candidate_v1.json and docs/SUPERBPE_V2_CANDIDATE.md.
2. Candidate objective J_A(c) formula, unweighted scalar ranking, tie-breaking,
   candidate filtering, reserve allocation (16 of 64), and fsum normalization.
3. Negative control contrast (fallback_weight=5).
4. Evaluation gates 1-5 (exact accounting, BpT retention >= 99%, fallback ratio <= 1.0,
   p95 span ratio <= 1.0, fragmentation increase <= 0.5 pp, aggregate reduction >= 10%).
5. Strict gate tripping and rejection enforcement on any violation, including missing
   counts or undefined strata which must block certification per protocol rules.
6. Adoption prerequisites (independent English and code validation coverage).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple
import unittest

from benchmarks import byte_fallback_analysis as b
from tests.test_byte_fallback_analysis import byte_model
from uniqtoken.byte_codec import ByteFallbackEngine as Bytes
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer


def load_candidate_protocol() -> Dict[str, Any]:
    """Load the machine-readable SuperBPE v2 candidate protocol specification JSON."""
    protocol_path = Path(__file__).resolve().parents[1] / "benchmarks" / "protocols" / "superbpe_v2_candidate_v1.json"
    with open(protocol_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_v2_candidate_gates(
    budget_evaluations: Dict[int, Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate candidate vs baseline across budgets and strata against protocol gates.

    Enforces strict protocol integrity rules:
    - Missing counts block certification rather than passing through defaults.
    - Zero runs in fragmentation categories produce an explicit not-applicable record.
    - Aggregate reductions are computed over shared strata only.
    - Any gate failure or missing required metric causes immediate rejection.

    Returns (passed, audit_report). If any gate fails on any budget or required stratum,
    passed is False and audit_report records the specific failure reason.
    """
    gates = protocol["gates"]
    min_bpt_ratio = gates["per_stratum_minimum_bpt_ratio"]
    max_fsb_ratio = gates["per_stratum_maximum_fallback_source_byte_ratio"]
    max_p95_ratio = gates["per_stratum_maximum_p95_fallback_span_ratio"]
    max_frag_increase_pp = gates["per_stratum_maximum_fragmentation_increase_percentage_points"]
    min_agg_reduction = gates["minimum_aggregate_fallback_source_byte_reduction"]
    min_budgets_with_reduction = gates["minimum_budgets_with_reduction"]

    report: Dict[str, Any] = {
        "budgets": {},
        "budgets_with_sufficient_reduction": 0,
        "violations": [],
    }

    budgets_meeting_reduction = 0

    for budget, data in sorted(budget_evaluations.items()):
        b_report: Dict[str, Any] = {
            "strata": {},
            "gate1_passed": True,
            "gate2_passed": True,
            "gate3_passed": True,
            "gate4_passed": True,
            "aggregate_reduction": None,
            "aggregate_passed": False,
        }

        # Gate 1: Exact accounting and integrity
        cand_model_info = data.get("candidate_model_info")
        if not isinstance(cand_model_info, dict):
            b_report["gate1_passed"] = False
            report["violations"].append(f"Budget {budget}: blocked, missing candidate_model_info")
            continue

        actual_vocab = cand_model_info.get("actual_vocab_size")
        if actual_vocab != budget:
            b_report["gate1_passed"] = False
            report["violations"].append(f"Budget {budget}: vocab size {actual_vocab} != budget {budget}")

        if "incomplete_prefix_additions" not in cand_model_info:
            b_report["gate1_passed"] = False
            report["violations"].append(f"Budget {budget}: blocked, missing incomplete_prefix_additions count")
        elif cand_model_info["incomplete_prefix_additions"] != 0:
            b_report["gate1_passed"] = False
            report["violations"].append(f"Budget {budget}: incomplete prefix additions detected")

        if cand_model_info.get("probabilities_finite") is not True:
            b_report["gate1_passed"] = False
            report["violations"].append(f"Budget {budget}: non-finite or missing probability verification")

        baseline_strata = data.get("baseline_strata")
        cand_strata = data.get("candidate_strata")

        if not baseline_strata or not cand_strata:
            report["violations"].append(f"Budget {budget}: missing baseline or candidate strata data")
            return False, report

        for stratum_name, base_s in sorted(baseline_strata.items()):
            if stratum_name not in cand_strata:
                b_report["gate2_passed"] = False
                report["violations"].append(f"Budget {budget}: stratum {stratum_name} missing in candidate")
                continue

            cand_s = cand_strata[stratum_name]
            stratum_report: Dict[str, Any] = {}

            # Strict key presence validation per protocol blocked_not_passed rule
            required_stratum_keys = (
                "bytes_per_token",
                "normalized_utf8_bytes",
                "fallback_source_bytes",
                "p95_fallback_span",
                "fragmentation",
            )
            missing_base = [k for k in required_stratum_keys if k not in base_s]
            missing_cand = [k for k in required_stratum_keys if k not in cand_s]
            if missing_base or missing_cand:
                b_report["gate2_passed"] = False
                b_report["gate3_passed"] = False
                report["violations"].append(
                    f"Budget {budget}, stratum {stratum_name}: blocked, missing required count(s) "
                    f"(baseline: {missing_base}, candidate: {missing_cand})"
                )
                continue

            # Gate 2: BpT retention
            base_bpt = base_s["bytes_per_token"]
            cand_bpt = cand_s["bytes_per_token"]
            if (
                base_bpt is None
                or cand_bpt is None
                or not math.isfinite(base_bpt)
                or not math.isfinite(cand_bpt)
                or base_bpt <= 0
                or cand_bpt <= 0
            ):
                b_report["gate2_passed"] = False
                report["violations"].append(f"Budget {budget}, stratum {stratum_name}: undefined/nonfinite BpT")
            else:
                bpt_ratio = cand_bpt / base_bpt
                stratum_report["bpt_ratio"] = bpt_ratio
                if bpt_ratio < min_bpt_ratio:
                    b_report["gate2_passed"] = False
                    report["violations"].append(
                        f"Budget {budget}, stratum {stratum_name}: BpT ratio {bpt_ratio:.4f} < {min_bpt_ratio}"
                    )

            # Gate 3: Fallback source-byte ratio & p95 span ratio
            base_bytes = base_s["normalized_utf8_bytes"]
            cand_bytes = cand_s["normalized_utf8_bytes"]
            base_fb = base_s["fallback_source_bytes"]
            cand_fb = cand_s["fallback_source_bytes"]

            if base_bytes <= 0 or cand_bytes <= 0:
                b_report["gate3_passed"] = False
                report["violations"].append(f"Budget {budget}, stratum {stratum_name}: non-positive source bytes")
            else:
                base_fsbf = base_fb / base_bytes
                cand_fsbf = cand_fb / cand_bytes
                if base_fsbf == 0.0:
                    if cand_fsbf != 0.0:
                        b_report["gate3_passed"] = False
                        report["violations"].append(
                            f"Budget {budget}, stratum {stratum_name}: baseline has 0 fallback but candidate has {cand_fsbf}"
                        )
                else:
                    fsb_ratio = cand_fsbf / base_fsbf
                    stratum_report["fsb_ratio"] = fsb_ratio
                    if fsb_ratio > max_fsb_ratio:
                        b_report["gate3_passed"] = False
                        report["violations"].append(
                            f"Budget {budget}, stratum {stratum_name}: fallback ratio {fsb_ratio:.4f} > {max_fsb_ratio}"
                        )

            # Gate 3: p95 span ratio
            base_p95 = base_s["p95_fallback_span"]
            cand_p95 = cand_s["p95_fallback_span"]
            if base_p95 is None:
                # Empty span histogram (known absence of fallback)
                if cand_p95 is not None:
                    b_report["gate3_passed"] = False
                    report["violations"].append(
                        f"Budget {budget}, stratum {stratum_name}: baseline empty span but candidate has p95={cand_p95}"
                    )
            elif cand_p95 is None:
                # Candidate eliminated fallback entirely: passes
                pass
            else:
                if base_p95 <= 0:
                    if cand_p95 > 0:
                        b_report["gate3_passed"] = False
                        report["violations"].append(
                            f"Budget {budget}, stratum {stratum_name}: p95 increased from {base_p95} to {cand_p95}"
                        )
                else:
                    p95_ratio = cand_p95 / base_p95
                    stratum_report["p95_ratio"] = p95_ratio
                    if p95_ratio > max_p95_ratio:
                        b_report["gate3_passed"] = False
                        report["violations"].append(
                            f"Budget {budget}, stratum {stratum_name}: p95 span ratio {p95_ratio:.4f} > {max_p95_ratio}"
                        )

            # Gate 4: Fragmentation split runs
            base_runs = base_s["fragmentation"]
            cand_runs = cand_s["fragmentation"]
            for cat in ("whitespace", "punctuation"):
                if cat not in base_runs or cat not in cand_runs:
                    b_report["gate4_passed"] = False
                    report["violations"].append(
                        f"Budget {budget}, stratum {stratum_name}: blocked, missing {cat} fragmentation data"
                    )
                    continue

                base_cat = base_runs[cat]
                cand_cat = cand_runs[cat]
                if (
                    "runs" not in base_cat
                    or "runs" not in cand_cat
                    or "split_runs" not in base_cat
                    or "split_runs" not in cand_cat
                ):
                    b_report["gate4_passed"] = False
                    report["violations"].append(
                        f"Budget {budget}, stratum {stratum_name}: blocked, missing {cat} run counts"
                    )
                    continue

                b_total = base_cat["runs"]
                c_total = cand_cat["runs"]

                if b_total == 0 or c_total == 0:
                    # Protocol: zero runs gives null and an explicitly not-applicable category with run counts;
                    # missing counts block certification; never replace null with zero.
                    stratum_report[f"frag_{cat}"] = {
                        "status": "not_applicable",
                        "baseline_runs": b_total,
                        "candidate_runs": c_total,
                        "baseline_split_runs": base_cat["split_runs"],
                        "candidate_split_runs": cand_cat["split_runs"],
                    }
                    continue

                b_pct = 100.0 * base_cat["split_runs"] / b_total
                c_pct = 100.0 * cand_cat["split_runs"] / c_total
                delta_pp = c_pct - b_pct
                stratum_report[f"frag_delta_{cat}"] = delta_pp
                if delta_pp > max_frag_increase_pp:
                    b_report["gate4_passed"] = False
                    report["violations"].append(
                        f"Budget {budget}, stratum {stratum_name}: {cat} fragmentation delta {delta_pp:.4f} pp > {max_frag_increase_pp} pp"
                    )

            b_report["strata"][stratum_name] = stratum_report

        # Gate 5: Aggregate fallback source-byte reduction over shared strata only
        shared = sorted(set(baseline_strata.keys()) & set(cand_strata.keys()))
        base_agg_fb = sum(
            baseline_strata[s]["fallback_source_bytes"] for s in shared if "fallback_source_bytes" in baseline_strata[s]
        )
        base_agg_bytes = sum(
            baseline_strata[s]["normalized_utf8_bytes"] for s in shared if "normalized_utf8_bytes" in baseline_strata[s]
        )
        cand_agg_fb = sum(
            cand_strata[s]["fallback_source_bytes"] for s in shared if "fallback_source_bytes" in cand_strata[s]
        )
        cand_agg_bytes = sum(
            cand_strata[s]["normalized_utf8_bytes"] for s in shared if "normalized_utf8_bytes" in cand_strata[s]
        )

        if base_agg_bytes > 0 and cand_agg_bytes > 0:
            if base_agg_fb == 0:
                # Baseline had zero fallback: relative reduction cannot be computed
                b_report["aggregate_reduction"] = 0.0
            else:
                base_agg_frac = base_agg_fb / base_agg_bytes
                cand_agg_frac = cand_agg_fb / cand_agg_bytes
                reduction = 1.0 - (cand_agg_frac / base_agg_frac)
                b_report["aggregate_reduction"] = reduction
                if reduction >= min_agg_reduction:
                    b_report["aggregate_passed"] = True
                    budgets_meeting_reduction += 1

        report["budgets"][budget] = b_report

    report["budgets_with_sufficient_reduction"] = budgets_meeting_reduction

    gate5_passed = budgets_meeting_reduction >= min_budgets_with_reduction
    if not gate5_passed:
        report["violations"].append(
            f"Gate 5: only {budgets_meeting_reduction} budgets met {min_agg_reduction * 100}% reduction (required {min_budgets_with_reduction})"
        )

    all_passed = len(report["violations"]) == 0 and gate5_passed
    return all_passed, report


class SuperBPEv2ProtocolIntegrityTests(unittest.TestCase):
    """Test suite for benchmarks/protocols/superbpe_v2_candidate_v1.json integrity."""

    def setUp(self) -> None:
        """Load candidate protocol definition before running each test."""
        self.protocol = load_candidate_protocol()

    def test_protocol_metadata_and_schema(self) -> None:
        """Verify protocol schema version, frozen manifest hash, and research integrity flags."""
        self.assertEqual(self.protocol["schema_version"], 1)
        self.assertEqual(
            self.protocol["status"],
            "proposal_only_blocked_on_evidence_review_and_independent_validation",
        )
        self.assertEqual(
            self.protocol["result_scope"],
            "predeclared_exploratory_extension_not_independent_confirmation",
        )
        self.assertEqual(self.protocol["issue"], 94)
        self.assertEqual(self.protocol["evidence_prs"], [121, 124, 127, 128, 129])
        self.assertEqual(
            self.protocol["frozen_manifest_sha256"],
            "2ad27746d9c139c8d7e814e89c977c9bd4410033d5c1b97b673ee1b893ece7ff",
        )
        self.assertEqual(self.protocol["normalization"], "NFKC_unicode_spaces_v1")
        self.assertEqual(self.protocol["test_access"], "forbidden_not_opened")
        self.assertEqual(self.protocol["splits"], ["train", "validation"])
        self.assertEqual(self.protocol["budgets"], [8192, 16384, 32768, 65536, 131072])
        self.assertEqual(
            self.protocol["special_tokens"],
            ["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"],
        )
        self.assertEqual(self.protocol["special_ids"], [0, 1, 2, 3])
        self.assertEqual(self.protocol["byte_leaves"], 256)
        self.assertEqual(self.protocol["rayon_threads"], 1)
        self.assertEqual(self.protocol["python_hash_seed"], "0")
        self.assertEqual(self.protocol["randomness"], "none")
        self.assertIsNone(self.protocol["seed"])
        self.assertFalse(self.protocol["production_objective_changed"])
        self.assertFalse(self.protocol["frozen_phase_a_b_c_changed"])
        self.assertEqual(self.protocol["downstream_claim"], "none")

    def test_candidate_hyperparameters_and_cohort(self) -> None:
        """Verify candidate reserve, atomic limit, negative control weight, and cohort members."""
        candidate_cfg = self.protocol["candidate"]
        self.assertEqual(candidate_cfg["reserve"], 64)
        self.assertEqual(candidate_cfg["atomic_limit"], 16)
        self.assertEqual(candidate_cfg["minimum_atomic_frequency"], 2)
        self.assertEqual(candidate_cfg["fallback_weight"], 0)
        self.assertEqual(candidate_cfg["max_subword_len"], 16)
        self.assertFalse(candidate_cfg["byte_prefix_admission"])
        self.assertEqual(self.protocol["negative_control_fallback_weight"], 5)
        self.assertEqual(
            self.protocol["cohort"],
            [
                "uniq_superbpe_r64",
                "atomic_recovery",
                "fallback_weighted_negative_control",
                "sp_unigram",
                "boundary_bpe",
            ],
        )

    def test_candidate_proposal_documentation_consistency(self) -> None:
        """Verify docs/SUPERBPE_V2_CANDIDATE.md matches the machine-readable protocol specification."""
        doc_path = Path(__file__).resolve().parents[1] / "docs" / "SUPERBPE_V2_CANDIDATE.md"
        self.assertTrue(doc_path.exists())
        content = doc_path.read_text(encoding="utf-8")
        self.assertIn("Status: proposal only", content)
        self.assertIn("#94", content)
        for pr_num in [121, 124, 127, 128, 129]:
            self.assertIn(f"#{pr_num}", content)
        self.assertIn("J_A(c) = f(c) * (sum_j log P(b_j) - log(f(c) / N_e))", content)
        self.assertIn(
            "J_C(a,b) = f(a,b) * (log P(a) + log P(b) - log(f(a,b) / N_pairs))",
            content,
        )
        self.assertIn("Admit at most 16 candidates", content)
        self.assertIn("held-out test set", content)


class CandidateObjectiveFormulaTests(unittest.TestCase):
    """Test suite for candidate objective J_A(c) formula, tie breaks, and normalization."""

    def test_candidate_objective_calculation_matches_unweighted_recovery(
        self,
    ) -> None:
        """Verify unweighted candidate recovery computes exact J_A(c) score."""
        model = byte_model()
        chunks = ["\u00e9\u093e\U0001f600"] * 5
        candidates = b.recovery_candidates(model, chunks, fallback_weight=0.0)

        # Count total training emissions
        total_tokens = sum(len(model.encode(c)) for c in chunks)

        for row in candidates:
            char = row["token"]
            freq = row["frequency"]
            utf8_bytes = char.encode("utf-8")
            byte_tokens = [Bytes.byte_to_token(bt) for bt in utf8_bytes]
            log_prob = math.log(freq / total_tokens)
            expected_score = freq * (math.fsum(model.vocab[bt] for bt in byte_tokens) - log_prob)
            self.assertAlmostEqual(row["score"], expected_score, places=9)
            self.assertEqual(row["utf8_bytes"], len(utf8_bytes))

    def test_negative_control_fallback_weight_penalty(self) -> None:
        """Verify that fallback_weight=5 applies an explicit penalty as a negative control."""
        model = byte_model()
        chunks = ["\u00e9\u093e\U0001f600"] * 5
        plain = {r["token"]: r for r in b.recovery_candidates(model, chunks, fallback_weight=0.0)}
        weighted = {r["token"]: r for r in b.recovery_candidates(model, chunks, fallback_weight=5.0)}
        self.assertEqual(set(plain), set(weighted))
        for token, plain_row in plain.items():
            w_row = weighted[token]
            penalty = 5.0 * plain_row["frequency"] * plain_row["utf8_bytes"]
            self.assertAlmostEqual(w_row["score"], plain_row["score"] - penalty)

    def test_candidate_filtering_and_admission_rules(self) -> None:
        """Verify candidate admission enforces frequency >= 2 and multi-byte UTF-8 scalars."""
        model = byte_model()
        # Singletons (frequency=1) must not be admitted
        singletons = ["\u00e9", "\u093e", "\U0001f600"]
        cands = b.recovery_candidates(model, singletons, fallback_weight=0.0)
        self.assertEqual(len(cands), 0)

        # Multi-frequency complete scalars are admitted
        multi = ["\u00e9\u093e\U0001f600"] * 3
        cands = b.recovery_candidates(model, multi, fallback_weight=0.0)
        admitted_tokens = {c["token"] for c in cands}
        self.assertEqual(admitted_tokens, {"\u00e9", "\u093e", "\U0001f600"})
        # No byte notations admitted
        for c in cands:
            self.assertFalse(c["token"].startswith("<0x"))
            self.assertGreaterEqual(c["utf8_bytes"], 2)

    def test_tie_breaking_order(self) -> None:
        """Verify production recovery_candidates breaks ties by descending frequency, then scalar text."""
        # Equal frequency and equal byte log-probs give equal scores; order falls to scalar text.
        chars = [chr(0x0918 - i) for i in range(5)]
        rows = b.recovery_candidates(byte_model(), chars * 2, fallback_weight=0.0)
        self.assertEqual([r["token"] for r in rows], sorted(chars))
        keys = [(r["score"], -r["frequency"], r["token"]) for r in rows]
        self.assertEqual(keys, sorted(keys))

    def test_atomic_limit_cap_and_id_accounting(self) -> None:
        """Verify that at most 16 candidates are admitted and IDs are appended sequentially."""
        model = byte_model()
        initial_vocab_size = len(model.vocab)
        # Create 20 distinct multi-byte characters repeated twice
        chars = [chr(0x0905 + i) for i in range(20)]
        chunks = chars * 2
        updated, selected = b.recover_characters(model, chunks, limit=16, fallback_weight=0.0)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len(updated.vocab), initial_vocab_size + 16)

        # IDs must be strictly appended sequentially
        for i, row in enumerate(selected):
            expected_id = initial_vocab_size + i
            self.assertEqual(updated.token_to_id[row["token"]], expected_id)
            self.assertEqual(updated.id_to_token[expected_id], row["token"])

        # Check probability normalization with fsum
        summed_prob = math.fsum(math.exp(lp) for lp in updated.vocab.values())
        self.assertAlmostEqual(summed_prob, 1.0, places=9)
        self.assertTrue(all(math.isfinite(lp) for lp in updated.vocab.values()))
        self.assertTrue(all(lp <= 0.0 for lp in updated.vocab.values()))

    def test_roundtrip_and_reload_parity(self) -> None:
        """Verify CustomTokenizer preserves roundtrip decode parity with admitted scalar entries."""
        model, _ = b.recover_characters(byte_model(), ["\u00e9\u093e\U0001f600"] * 3, limit=16)
        tokenizer = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
        test_text = "test \u00e9 \u093e \U0001f600 text"
        ids = tokenizer.encode_to_ids(test_text)
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, test_text)


class CandidateGateEvaluationTests(unittest.TestCase):
    """Test suite for candidate gate evaluations (Gates 1 - 5) and rejection tripping."""

    def setUp(self) -> None:
        """Load candidate protocol definition before running each test."""
        self.protocol = load_candidate_protocol()

    def _create_passing_fixture(self) -> Dict[int, Dict[str, Any]]:
        """Construct a complete, passing synthetic benchmark evaluation fixture across all 5 budgets."""
        fixture: Dict[int, Dict[str, Any]] = {}
        for budget in (8192, 16384, 32768, 65536, 131072):
            fixture[budget] = {
                "candidate_model_info": {
                    "actual_vocab_size": budget,
                    "incomplete_prefix_additions": 0,
                    "probabilities_finite": True,
                },
                "baseline_strata": {
                    "hi": {
                        "bytes_per_token": 2.0,
                        "normalized_utf8_bytes": 10000,
                        "fallback_source_bytes": 1000,
                        "p95_fallback_span": 3,
                        "fragmentation": {
                            "whitespace": {"runs": 100, "split_runs": 10},
                            "punctuation": {"runs": 100, "split_runs": 15},
                        },
                    },
                    "ar": {
                        "bytes_per_token": 2.5,
                        "normalized_utf8_bytes": 10000,
                        "fallback_source_bytes": 800,
                        "p95_fallback_span": 2,
                        "fragmentation": {
                            "whitespace": {"runs": 80, "split_runs": 5},
                            "punctuation": {"runs": 90, "split_runs": 10},
                        },
                    },
                },
                "candidate_strata": {
                    "hi": {
                        # BpT ratio = 1.99 / 2.0 = 0.995 >= 0.99 (passes Gate 2)
                        "bytes_per_token": 1.99,
                        "normalized_utf8_bytes": 10000,
                        # Fallback ratio = 850 / 1000 = 0.85 <= 1.0 (passes Gate 3)
                        "fallback_source_bytes": 850,
                        "p95_fallback_span": 3,  # 3 / 3 = 1.0 <= 1.0
                        "fragmentation": {
                            # 10.2% - 10.0% = 0.2 pp <= 0.5 pp (passes Gate 4)
                            "whitespace": {"runs": 100, "split_runs": 10},
                            "punctuation": {"runs": 100, "split_runs": 15},
                        },
                    },
                    "ar": {
                        "bytes_per_token": 2.49,
                        "normalized_utf8_bytes": 10000,
                        "fallback_source_bytes": 700,
                        "p95_fallback_span": 2,
                        "fragmentation": {
                            "whitespace": {"runs": 80, "split_runs": 5},
                            "punctuation": {"runs": 90, "split_runs": 10},
                        },
                    },
                },
            }
        return fixture

    def test_passing_evaluations_meet_all_gates(self) -> None:
        """Verify that a compliant candidate meeting all criteria passes all evaluation gates."""
        fixture = self._create_passing_fixture()
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertTrue(passed, f"Violations: {report['violations']}")
        self.assertEqual(len(report["violations"]), 0)
        self.assertGreaterEqual(report["budgets_with_sufficient_reduction"], 2)

    def test_gate1_violation_trips_rejection_on_vocab_size_mismatch(self) -> None:
        """Verify that any vocabulary size mismatch against target budget rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Alter actual_vocab_size for 8192 to 8191
        fixture[8192]["candidate_model_info"]["actual_vocab_size"] = 8191
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("vocab size 8191" in v for v in report["violations"]))

    def test_gate1_violation_trips_rejection_on_incomplete_prefixes(self) -> None:
        """Verify that admitting incomplete UTF-8 prefixes strictly rejects the candidate."""
        fixture = self._create_passing_fixture()
        fixture[8192]["candidate_model_info"]["incomplete_prefix_additions"] = 1
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("incomplete prefix additions" in v for v in report["violations"]))

    def test_gate1_violation_trips_on_missing_probabilities_finite(self) -> None:
        """Verify that omitting probabilities_finite verification blocks Gate 1 per protocol rules."""
        fixture = self._create_passing_fixture()
        del fixture[8192]["candidate_model_info"]["probabilities_finite"]
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("probability verification" in v for v in report["violations"]))

    def test_gate2_violation_trips_rejection_on_bpt_loss(self) -> None:
        """Verify that exceeding the 1% BpT regression threshold strictly rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Lower BpT below 99% retention: 1.95 / 2.0 = 0.975 < 0.99
        fixture[8192]["candidate_strata"]["hi"]["bytes_per_token"] = 1.95
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("BpT ratio" in v for v in report["violations"]))

    def test_gate2_violation_trips_rejection_on_missing_stratum(self) -> None:
        """Verify that omitting a required stratum blocks certification rather than passing."""
        fixture = self._create_passing_fixture()
        del fixture[8192]["candidate_strata"]["hi"]
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("missing in candidate" in v for v in report["violations"]))

    def test_gate3_violation_trips_rejection_on_fallback_increase(self) -> None:
        """Verify that increasing fallback bytes per source byte strictly rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Candidate fallback higher than baseline
        fixture[8192]["candidate_strata"]["hi"]["fallback_source_bytes"] = 1100  # 1100/1000 = 1.1 > 1.0
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("fallback ratio" in v for v in report["violations"]))

    def test_gate3_violation_trips_rejection_on_zero_fallback_baseline_violation(
        self,
    ) -> None:
        """Verify that zero baseline fallback requires zero candidate fallback."""
        fixture = self._create_passing_fixture()
        # Baseline has zero fallback, candidate has non-zero
        fixture[8192]["baseline_strata"]["hi"]["fallback_source_bytes"] = 0
        fixture[8192]["candidate_strata"]["hi"]["fallback_source_bytes"] = 5
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("baseline has 0 fallback" in v for v in report["violations"]))

    def test_gate3_violation_trips_rejection_on_p95_span_increase(self) -> None:
        """Verify that increasing p95 fallback span strictly rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Candidate p95 span = 4, baseline = 3 (4 / 3 = 1.33 > 1.0)
        fixture[8192]["candidate_strata"]["hi"]["p95_fallback_span"] = 4
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("p95 span ratio" in v for v in report["violations"]))

    def test_gate3_violation_trips_on_missing_fallback_counts(self) -> None:
        """Verify that missing fallback_source_bytes count blocks certification."""
        fixture = self._create_passing_fixture()
        del fixture[8192]["candidate_strata"]["hi"]["fallback_source_bytes"]
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("missing required count" in v for v in report["violations"]))

    def test_gate3_violation_trips_on_missing_p95_span_counts(self) -> None:
        """Verify that missing p95_fallback_span count blocks certification."""
        fixture = self._create_passing_fixture()
        del fixture[8192]["candidate_strata"]["hi"]["p95_fallback_span"]
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("missing required count" in v for v in report["violations"]))

    def test_gate4_violation_trips_rejection_on_fragmentation_increase(
        self,
    ) -> None:
        """Verify that fragmentation increase exceeding 0.5 pp strictly rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Increase split runs by 1.0 pp (from 10% to 11% with 100 runs)
        fixture[8192]["candidate_strata"]["hi"]["fragmentation"]["whitespace"]["split_runs"] = 11
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("fragmentation delta" in v for v in report["violations"]))

    def test_gate4_violation_trips_on_missing_category_runs(self) -> None:
        """Verify that omitting category run counts blocks certification."""
        fixture = self._create_passing_fixture()
        del fixture[8192]["candidate_strata"]["hi"]["fragmentation"]["whitespace"]["runs"]
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertTrue(any("blocked, missing whitespace run counts" in v for v in report["violations"]))

    def test_gate4_zero_runs_produces_explicit_not_applicable_record(
        self,
    ) -> None:
        """Verify that zero runs produces an explicit not-applicable record rather than silent omission."""
        fixture = self._create_passing_fixture()
        fixture[8192]["candidate_strata"]["hi"]["fragmentation"]["whitespace"] = {
            "runs": 0,
            "split_runs": 0,
        }
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        # Should record explicit not_applicable entry
        hi_frag = report["budgets"][8192]["strata"]["hi"]["frag_whitespace"]
        self.assertEqual(hi_frag["status"], "not_applicable")
        self.assertEqual(hi_frag["candidate_runs"], 0)

    def test_gate5_violation_trips_rejection_on_insufficient_budgets_with_reduction(
        self,
    ) -> None:
        """Verify that failing to reach 10% aggregate reduction in at least 2 budgets rejects the candidate."""
        fixture = self._create_passing_fixture()
        # Set candidate fallback equal to baseline across 4 budgets (reduction = 0%)
        for budget in (8192, 16384, 32768, 65536):
            fixture[budget]["candidate_strata"]["hi"]["fallback_source_bytes"] = fixture[budget]["baseline_strata"][
                "hi"
            ]["fallback_source_bytes"]
            fixture[budget]["candidate_strata"]["ar"]["fallback_source_bytes"] = fixture[budget]["baseline_strata"][
                "ar"
            ]["fallback_source_bytes"]
        # Only 1 budget (131072) achieves >= 10% reduction, but 2 are required
        passed, report = evaluate_v2_candidate_gates(fixture, self.protocol)
        self.assertFalse(passed)
        self.assertEqual(report["budgets_with_sufficient_reduction"], 1)
        self.assertTrue(any("Gate 5" in v for v in report["violations"]))

    def test_adoption_requires_independent_english_and_code_validation(
        self,
    ) -> None:
        """Verify that independent validation of English and 10 programming languages is required."""
        required_independent = self.protocol["gates"]["required_independent_validation"]
        self.assertEqual(set(required_independent), {"english", "code"})
        required_langs = set(self.protocol["gates"]["required_independent_languages"])
        expected_langs = {
            "en",
            "c",
            "cpp",
            "go",
            "java",
            "javascript",
            "python",
            "rust",
            "sql",
            "typescript",
        }
        self.assertEqual(required_langs, expected_langs)


if __name__ == "__main__":
    unittest.main()

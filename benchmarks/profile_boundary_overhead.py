"""Paired public-API measurements of two Python guards, plus boundary diagnostics.

The Python-only probes do not measure Rust UTF-8 access or native materialization.
Independent timings are not an additive decomposition of end-to-end latency.
Only embedded synthetic fixtures are used; no research data is opened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
import unicodedata
from unittest.mock import patch

from benchmarks.profile_hot_paths import (
    FIXTURES,
    ROOT,
    git_value,
    make_tokenizer as make_benchmark_tokenizer,
    native_file,
    reference_raw_spans,
    sha256,
    tool_version,
    workload_cases,
)
from benchmarks.profile_residual_native import peak_rss
from uniqtoken import _native
from uniqtoken.tokenizer import CustomTokenizer
import uniqtoken.tokenizer as tokenizer_module

optimized_requires_python_security = CustomTokenizer._requires_python_security
NATIVE_IDS = ("rust_encode_text_native_ids", "rust_encode_text_native_ids_batch")
ITERATIONS = 5


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def source_sha256(filename):
    """Hash committed blobs so checkout newline conversion cannot alter receipts."""
    blob = subprocess.check_output(["git", "show", f"HEAD:{filename}"], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def baseline_native_text_supported(text):
    """The pre-PR helper, including its surrogate warning."""
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        logging.getLogger("uniqtoken.native").warning(
            "lone surrogate cannot cross the UTF-8 native boundary; using Python implementation"
        )
        return False
    return True


def baseline_requires_python_security(text):
    return (
        not baseline_native_text_supported(text)
        or "\ue000" in text
        or "\ue001" in text
        or "<|" in unicodedata.normalize("NFKC", text)
    )


@contextmanager
def implementation(variant):
    """Restore pre-PR checks inside the public call path, outside timed regions."""
    if variant == "optimized":
        yield
        return
    if variant != "baseline":
        raise ValueError(f"unknown implementation: {variant}")
    current = _native.native_text_supported
    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name.startswith("uniqtoken") and getattr(module, "native_text_supported", None) is current:
                stack.enter_context(patch.object(module, "native_text_supported", baseline_native_text_supported))
        stack.enter_context(
            patch.object(CustomTokenizer, "_requires_python_security", staticmethod(baseline_requires_python_security))
        )
        yield


def require_native(tok):
    native = tokenizer_module._native_core
    if (
        native is None
        or any(not callable(getattr(native, name, None)) for name in NATIVE_IDS)
        or tok._native_pipeline_kwargs() is None
        or tok.model._get_rust_trie() is None
    ):
        raise RuntimeError("native IDs APIs and a compatible native tokenizer are required")
    return native


def observed_native_calls(tok, call):
    """Count actual native API entries in a separate, untimed invocation."""
    native = require_native(tok)
    counts = Counter()
    with ExitStack() as stack:
        for name in NATIVE_IDS:
            original = getattr(native, name)

            def counted(*args, _name=name, _original=original, **kwargs):
                counts[_name] += 1
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(native, name, counted))
        result = call()
    return dict(counts), result


def capture(call):
    try:
        return {"value": call()}
    except Exception as error:
        return {"error": type(error).__name__, "args": repr(error.args)}


def snapshot(tok, text, options):
    calls = {
        "tokens": lambda: tok.encode(text, **options),
        "ids": lambda: tok.encode_to_ids(text, **options),
        "offsets": lambda: [(t.text, t.id, t.raw_span) for t in tok.encode_with_offsets(text, **options)],
        "batch_tokens": lambda: tok.encode_batch([text, text], **options),
        "batch_ids": lambda: tok.encode_to_ids_batch([text, text], **options),
        "batch_offsets": lambda: [
            [(t.text, t.id, t.raw_span) for t in row] for row in tok.encode_with_offsets_batch([text, text], **options)
        ],
        "decode_ids": lambda: tok.decode(tok.encode_to_ids(text, **options)),
        "decode_tokens": lambda: tok.decode_tokens(tok.encode(text, **options)),
        "decode_batch": lambda: tok.decode_batch(tok.encode_to_ids_batch([text, text], **options)),
        "invalid_ids": lambda: tok.decode([-1, tok.vocab_size + 100]),
    }
    return {name: capture(call) for name, call in calls.items()}


def run_exact_parity_gate(tok, fixtures=FIXTURES):
    cases = [(name, text, {}) for name, text in fixtures.items()]
    security_texts = [
        "literal <|unk|> marker",
        "compatibility \ufe64\uff5cunk\uff5c\uff1e marker",
        "combining <\u0301|unk|> marker",
        "private \ue000 and \ue001",
        "surrogate \ud800",
        "emoji \U0001f469\u200d\U0001f4bb",
    ]
    for index, text in enumerate(security_texts):
        for action in ("escape", "raise", "ignore"):
            cases.append((f"security_{index}_{action}", text, {"disallowed_special_action": action}))
        cases.append((f"security_{index}_allowed", text, {"allowed_special": "all"}))
    verified = []
    for name, text, options in cases:
        with implementation("baseline"):
            before = snapshot(tok, text, options)
        after = snapshot(tok, text, options)
        if before != after:
            raise AssertionError(f"baseline/optimized API parity failed: {name}")
        if name in fixtures:
            if any("error" in value for key, value in after.items() if key != "invalid_ids"):
                raise AssertionError(f"ordinary fixture raised: {name}")
            tokens, ids = after["tokens"]["value"], after["ids"]["value"]
            offsets = after["offsets"]["value"]
            if [(t, i) for t, i, _ in offsets] != list(zip(tokens, ids)):
                raise AssertionError(f"offset token/ID mismatch: {name}")
            if [span for _, _, span in offsets] != reference_raw_spans(tok, text, tokens):
                raise AssertionError(f"independent raw-span oracle failed: {name}")
            if after["decode_ids"] != after["decode_tokens"]:
                raise AssertionError(f"decode mismatch: {name}")
            for single, batch in (("tokens", "batch_tokens"), ("ids", "batch_ids"), ("offsets", "batch_offsets")):
                if after[batch]["value"] != [after[single]["value"]] * 2:
                    raise AssertionError(f"single/batch mismatch: {name}/{single}")
        verified.append({"name": name, "input_sha256": digest(text), "output_sha256": digest(after)})
    return verified


def timed(call, iterations=1):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        call()
    return (time.perf_counter_ns() - start) / iterations / 1e6


def timing_summary(samples):
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("timing samples must be positive and finite")
    return {
        "samples_ms": samples,
        "p50_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
    }


def python_memory(call):
    """Retained/peak traced Python heap; excludes Rust allocations and RSS."""
    if tracemalloc.is_tracing():
        raise RuntimeError("a separate tracemalloc session is required")
    tracemalloc.start()
    try:
        result = call()
        retained, peak = tracemalloc.get_traced_memory()
        return {"retained_bytes": retained, "peak_bytes": peak, "scope": "Python traced heap only"}
    finally:
        tracemalloc.stop()


def probe(description, call, warmup, reps):
    for _ in range(warmup):
        call()
    return {
        "description": description,
        "scope": "Python-only diagnostic",
        "native_calls": 0,
        "copied_native_bytes": None,
        "latency": timing_summary([timed(call) for _ in range(reps)]),
        "memory": python_memory(call),
    }


def run_crossing_1_string_utf8_access(warmup=3, reps=11):
    texts = [FIXTURES["medium"]] * 32
    return {
        name: probe(
            "Python surrogate-support check; no Rust UTF-8 access", lambda: [helper(t) for t in texts], warmup, reps
        )
        for name, helper in (("baseline", baseline_native_text_supported), ("optimized", _native.native_text_supported))
    }


def run_crossing_2_token_ids_to_containers(warmup=3, reps=11):
    return {
        f"ids_{size}": probe(
            "Shallow Python list copy of existing int references; no Rust integer conversion",
            lambda ids=list(range(size)): list(ids),
            warmup,
            reps,
        )
        for size in (64, 512, 4096)
    }


def run_crossing_3_batch_output_construction(warmup=3, reps=11):
    ids = list(range(512))
    return {
        f"batch_{size}": probe(
            "Python nested lists of existing int references; no native output materialization",
            lambda size=size: [list(ids) for _ in range(size)],
            warmup,
            reps,
        )
        for size in (1, 8, 32, 128)
    }


def run_crossing_4_python_object_creation(warmup=3, reps=11):
    return {
        "ints": probe("Python int creation", lambda: [int(str(i)) for i in range(1000)], warmup, reps),
        "strings": probe("Python string creation", lambda: [str(i) for i in range(1000)], warmup, reps),
    }


def run_crossing_5_repeated_ffi_crossings(tok, warmup=3, reps=11):
    texts = [FIXTURES["medium"]] * 32
    records, outputs = {}, []
    for name, call, expected in (
        ("iterative", lambda: [tok.encode_to_ids(t) for t in texts], {NATIVE_IDS[0]: len(texts)}),
        ("batch", lambda: tok.encode_to_ids_batch(texts), {NATIVE_IDS[1]: 1}),
    ):
        counts, result = observed_native_calls(tok, call)
        if counts != expected:
            raise AssertionError(f"unexpected native path: {name}: {counts}")
        outputs.append(result)
        for _ in range(warmup):
            call()
        records[name] = {
            "native_calls": counts,
            "latency": timing_summary([timed(call) for _ in range(reps)]),
            "memory": python_memory(call),
            "copied_native_bytes": None,
        }
    if outputs[0] != outputs[1]:
        raise AssertionError("iterative/fused native IDs differ")
    return records


def measure_boundary_diagnostics(tok, texts, warmup=3, reps=11):
    calls = {
        "public_ids_batch": lambda: tok.encode_to_ids_batch(texts),
        "python_security_checks": lambda: [tok._requires_python_security(text) for text in texts],
    }
    result = {"scope": "independent operations; timings are not additive stages"}
    for name, call in calls.items():
        for _ in range(warmup):
            call()
        result[name] = timing_summary([timed(call) for _ in range(reps)])
    result.update(native_compute_ms=None, native_materialization_ms=None)
    return result


def paired_interval(ratios):
    rng = random.Random(0)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(1000))
    return [bootstrap[24], bootstrap[974]]


def run_macrobenchmark_matrix(tok, warmup=3, reps=11, cases=None):
    records = []
    for name, texts in workload_cases() if cases is None else cases:
        single = name.endswith("_single")
        call = (lambda: tok.encode_to_ids(texts[0])) if single else (lambda: tok.encode_to_ids_batch(texts))
        expected = {NATIVE_IDS[0 if single else 1]: 1}
        normalized_bytes = sum(len(tok.normalizer.normalize(t).encode("utf-8")) for t in texts)
        samples = {variant: [] for variant in ("baseline", "optimized")}
        memory, counts = {}, {}
        for variant in samples:
            with implementation(variant):
                counts[variant], _ = observed_native_calls(tok, call)
                if counts[variant] != expected:
                    raise AssertionError(f"fallback in macro cell: {name}/{variant}")
                for _ in range(warmup):
                    for _ in range(ITERATIONS):
                        call()
                memory[variant] = python_memory(call)
        for repetition in range(reps):
            order = ("baseline", "optimized") if repetition % 2 == 0 else ("optimized", "baseline")
            for variant in order:
                with implementation(variant):
                    samples[variant].append(timed(call, ITERATIONS))
        ratios = [before / after for before, after in zip(samples["baseline"], samples["optimized"])]
        summary = {variant: timing_summary(values) for variant, values in samples.items()}
        for variant in samples:
            summary[variant]["normalized_MB_per_s"] = normalized_bytes / summary[variant]["p50_ms"] / 1000
        records.append(
            {
                "workload": name,
                "batch_size": len(texts),
                "normalized_utf8_bytes": normalized_bytes,
                "latency": summary,
                "paired_speed_ratio": statistics.median(ratios),
                "paired_ratio_95pct_bootstrap_interval": paired_interval(ratios),
                "native_calls": counts,
                "python_memory": memory,
                "output_sha256": digest(call()),
            }
        )
    return records


def generate_markdown_report(payload):
    metadata = payload["metadata"]
    lines = [
        "# Python guard optimization and boundary diagnostics (#101)",
        "",
        "## Scope and method",
        "",
        "This is a synthetic engineering diagnostic, not an isolated FFI-stage profile. The baseline restores the",
        "two pre-PR Python checks inside the public API. Both variants use the same model, native binary, inputs,",
        "and configuration; no extra precheck is added outside encode. Paired trials alternate execution order.",
        "Memory is measured separately with tracemalloc and covers only the Python traced heap, not Rust allocations.",
        "",
        f"- Source commit: `{metadata['git_commit']}`; clean tracked tree: `{metadata['tracked_tree_clean']}`.",
        f"- Platform: `{metadata['platform']}`; Python `{metadata['python']}`; Unicode `{metadata['unicode']}`.",
        f"- Native SHA-256: `{metadata['native_sha256']}`; declared build mode: `{metadata['build_mode']}`.",
        f"- Rayon threads: {metadata['rayon_threads']}; warmup: {metadata['warmup']}; repetitions: {metadata['repetitions']};",
        f"  calls per timed public-API trial: {metadata['iterations']}.",
        "- Fixtures: five embedded text workloads, each single and batch of 32; normalized UTF-8 decimal MB/s.",
        "- Intervals bootstrap the paired median ratios within one process; they do not establish cross-machine effects.",
        "- Reproduce with a fresh release extension and committed source:",
        "  `python -m benchmarks.profile_boundary_overhead --output <new-directory> --threads 1 --warmup 3 --repetitions 11 --build-mode release`.",
        "",
        "## Public IDs API results",
        "",
        "| Workload | Before MB/s | After MB/s | Before p50/p95 ms | After p50/p95 ms | Paired speed ratio [95% interval] |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["macrobenchmarks"]:
        before, after = (row["latency"][variant] for variant in ("baseline", "optimized"))
        low, high = row["paired_ratio_95pct_bootstrap_interval"]
        lines.append(
            f"| {row['workload']} | {before['normalized_MB_per_s']:.3f} | {after['normalized_MB_per_s']:.3f} | "
            f"{before['p50_ms']:.4f}/{before['p95_ms']:.4f} | {after['p50_ms']:.4f}/{after['p95_ms']:.4f} | "
            f"{row['paired_speed_ratio']:.3f} [{low:.3f}, {high:.3f}] |"
        )
    lines += [
        "",
        "Ratios above 1 favor the optimized checks. All cells are reported, including regressions and uncertain",
        "intervals. These comparisons change both checks together; they do not attribute gains to either check alone.",
        "",
        "## Boundary diagnostics and parity",
        "",
        "The first four probe groups are Python-only operations: surrogate scanning, shallow copies of existing",
        "integer references, nested-list creation, and Python object creation. They do not time Rust UTF-8 borrowing",
        "or u32-to-Python conversion. Copied native bytes and Rust allocation volume remain unmeasured.",
        "",
        "The fifth group verifies 32 actual single native IDs calls versus one fused native batch IDs call, then",
        "times those public operations without instrumentation. Their output IDs must match exactly.",
        "",
        f"Baseline/optimized parity passed for {len(payload['parity'])} cases covering tokens, IDs, raw offsets, batch",
        "outputs, decode, invalid IDs, compatibility markers, private-use escapes, surrogates, and security policies.",
        "Ordinary fixture spans are also checked against an independent normalized-text alignment oracle.",
        "",
        "Independent public-batch and Python-check timings are diagnostic operations, not additive stages.",
        "Native compute and materialization fields are null because they have not been isolated.",
        "",
        "## Remaining work",
        "",
        "Issue #101 remains partially addressed: isolated native input access, materialization, copied bytes, and",
        "allocation-volume attribution still require native instrumentation. No universal performance claim follows",
        "from these synthetic measurements. Frozen research artifacts and release configuration are unchanged.",
        "",
    ]
    return "\n".join(lines)


def publish(payload, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_bytes((json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode())
    (output / "REPORT.md").write_bytes(generate_markdown_report(payload).encode())
    manifest = {
        "schema_version": 2,
        "issue": 101,
        "published_files": {name: sha256(output / name) for name in ("results.json", "REPORT.md")},
    }
    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; existing evidence is never overwritten"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--build-mode", choices=("release", "debug"), default="release")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    if args.warmup < 0 or args.repetitions < 3 or args.threads < 1:
        parser.error("warmup >= 0, repetitions >= 3, and threads >= 1 are required")
    if git_value("status", "--porcelain", "--untracked-files=no"):
        parser.error("commit tracked source changes before recording evidence")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    logging.getLogger("uniqtoken").setLevel(logging.ERROR)
    tok = make_benchmark_tokenizer()
    require_native(tok)
    parity = run_exact_parity_gate(tok)
    metadata = {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_tree": git_value("rev-parse", "HEAD^{tree}"),
        "tracked_tree_clean": True,
        "source_sha256": {
            name: source_sha256(name)
            for name in (
                "benchmarks/profile_boundary_overhead.py",
                "benchmarks/profile_hot_paths.py",
                "benchmarks/profile_residual_native.py",
                "uniqtoken/_native.py",
                "uniqtoken/tokenizer.py",
            )
        },
        "native_sha256": sha256(Path(native_file())),
        "native_filename": Path(native_file()).name,
        "model_sha256": digest(sorted((t, score, tok.model.token_to_id[t]) for t, score in tok.model.vocab.items())),
        "fixture_sha256": digest(FIXTURES),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": None,
        "ram_bytes": None,
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "rustc": tool_version("rustc", "--version"),
        "build_mode": args.build_mode,
        "rayon_threads": args.threads,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "iterations": ITERATIONS,
        "utc_time": datetime.now(timezone.utc).isoformat(),
    }
    probes = {
        "python_support_checks": run_crossing_1_string_utf8_access(args.warmup, args.repetitions),
        "python_list_copies": run_crossing_2_token_ids_to_containers(args.warmup, args.repetitions),
        "python_nested_lists": run_crossing_3_batch_output_construction(args.warmup, args.repetitions),
        "python_objects": run_crossing_4_python_object_creation(args.warmup, args.repetitions),
        "native_single_vs_batch": run_crossing_5_repeated_ffi_crossings(tok, args.warmup, args.repetitions),
    }
    payload = {
        "schema_version": 2,
        "issue": 101,
        "issue_completion": "partial",
        "metadata": metadata,
        "parity": parity,
        "probes": probes,
        "independent_diagnostics": measure_boundary_diagnostics(
            tok, [FIXTURES["medium"]] * 32, args.warmup, args.repetitions
        ),
        "macrobenchmarks": run_macrobenchmark_matrix(tok, args.warmup, args.repetitions),
        "process_peak_rss_bytes": peak_rss(),
    }
    publish(payload, args.output)
    print(f"Wrote verified synthetic diagnostics to {args.output}")


if __name__ == "__main__":
    main()

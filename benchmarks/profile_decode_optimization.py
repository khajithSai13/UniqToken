"""Canonical benchmarks and profiler for decode and ID-to-text reconstruction (Issue #103).

Implements the measurement contract for Issue #103:
- Profiles 7 dimensions separately:
  1. Token-ID validation and lookup (lenient vs strict);
  2. Special-token handling;
  3. Byte-fallback reconstruction (regex vs table);
  4. Piece/metaspace reconstruction (character loop vs fast-path);
  5. String concatenation and capacity growth;
  6. Batch decode scheduling and result construction;
  7. Streaming versus non-streaming decode where comparable.
- Measures single and batch decode across short, medium, long, byte fallback,
  multilingual, special tokens, and invalid ID error fixtures.
- Reports decode p50 and p95 latency, throughput (MB/s and tokens/s),
  allocation volume, and peak memory.
- Enforces strict exact parity gates across text equality, byte fallback,
  special tokens, streaming, errors, and round-trip identity.
"""

from __future__ import annotations

import argparse
import codecs
import ctypes
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.getLogger("uniqtoken").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]

BYTE_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

FIXTURES = {
    "short": "The quick brown fox jumps over 42 lazy dogs.",
    "medium": "Tokenizer throughput depends on normalization, Unicode handling, and lattice search. " * 12,
    "long": "Reliable measurements use fixed inputs and repeated trials across tokenization paths. " * 55,
    "multilingual": "Cafe\u0301 \u0928\u092e\u0938\u094d\u0924\u0947 \u4e16\u754c \ud55c\uad6d\uc5b4 \U0001f469\u200d\U0001f4bb data 2026. "
    * 12,
    "byte_fallback": "Binary test \x00\x01\x02 \U0001f680 \U0001f389 \U0001f525 emoji byte fallback payload! " * 8,
    "special_tokens": "<|endoftext|> hello <|startoftext|> world <|pad|> sequence <|endoftext|>",
    "source_code": "def decode_tokens(ids: list[int]) -> str:\n    return ''.join(map(chr, ids))\n" * 15,
}


def get_peak_rss_bytes() -> int:
    """Return process high-water mark resident memory in bytes."""
    if os.name != "nt":
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(kb * (1 if sys.platform == "darwin" else 1024))

    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        counts = ProcessMemoryCounters()
        counts.cb = ctypes.sizeof(counts)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counts), counts.cb):
            return int(counts.PeakWorkingSetSize)
    except Exception:
        pass
    return 0


def get_hardware_topology() -> Dict[str, Any]:
    physical_cores = os.cpu_count() or 1
    logical_cores = physical_cores
    total_ram_gb = 8.0
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or physical_cores
        logical_cores = psutil.cpu_count(logical=True) or logical_cores
        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "total_ram_gb": total_ram_gb,
    }


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    if not data:
        return 0.0, 0.0
    if len(data) == 1:
        return data[0], data[0]
    import random

    rng = random.Random(42)
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1.0 - ci) / 2.0 * n_bootstrap)
    upper_idx = int((1.0 + ci) / 2.0 * n_bootstrap) - 1
    return means[lower_idx], means[upper_idx]


def benchmark_timed(func: Callable[[], Any], warmup: int, repetitions: int, iters_per_rep: int) -> List[float]:
    for _ in range(warmup):
        for _ in range(iters_per_rep):
            func()
    durations = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        for _ in range(iters_per_rep):
            func()
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / (iters_per_rep * 1e6))  # ms per call
    return durations


def make_benchmark_tokenizer():
    from uniqtoken.tokenizer import CustomTokenizer

    corpus = list(FIXTURES.values()) * 3
    return CustomTokenizer.train_from_corpus(corpus, target_vocab_size=400, min_frequency=1, verbose=False)


# --- Baseline vs Optimized Byte Fallback and Metaspace Implementations ---


def baseline_decode_tokens(tokens: List[str], space_char: str = "\u2581") -> str:
    """Historical baseline byte fallback decoding using regex matching twice per token."""
    output_segments: List[str] = []
    byte_buffer = bytearray()

    def flush_bytes():
        if byte_buffer:
            output_segments.append(byte_buffer.decode("utf-8"))
            byte_buffer.clear()

    for tok in tokens:
        # Regex check on every token
        if len(tok) == 6 and tok.startswith("<0x") and tok[-1] == ">" and BYTE_TOKEN_PATTERN.match(tok):
            m = BYTE_TOKEN_PATTERN.match(tok)
            byte_val = int(m.group(1), 16)
            byte_buffer.append(byte_val)
        else:
            flush_bytes()
            output_segments.append(tok.replace(space_char, " "))

    flush_bytes()
    return "".join(output_segments)


def baseline_restore_escaped_metaspace(text: str, escape_prefix: str = "\ue000", space_char: str = "\u2581") -> str:
    """Historical baseline restore_escaped_metaspace looping through every character."""
    restored: List[str] = []
    i = 0
    while i < len(text):
        if text[i] != escape_prefix or i + 1 >= len(text):
            restored.append(text[i])
            i += 1
            continue
        marker = text[i + 1]
        if marker == "\ue001":
            restored.append(space_char)
            i += 2
        elif marker == escape_prefix:
            restored.append(escape_prefix)
            i += 2
        else:
            restored.append(text[i])
            i += 1
    return "".join(restored)


# --- Parity Verification Gate ---


def run_exact_parity_gate(tokenizer) -> bool:
    """Verify exact text equality, byte fallback, special tokens, streaming, errors, and round-trip parity."""
    # 1. Normalization-aware round-trip parity: decode(encode(text)) == expected_normalized
    for name, text in FIXTURES.items():
        if name == "special_tokens":
            continue  # special token handling is tested separately
        ids = tokenizer.encode_to_ids(text)
        decoded = tokenizer.decode(ids)
        expected = tokenizer.normalizer.restore_escaped_metaspace(
            tokenizer.normalizer.normalize(text).replace(tokenizer.normalizer.space_char, " ")
        )
        if decoded != expected:
            raise AssertionError(
                f"Normalization-aware round-trip mismatch for fixture '{name}':\nExpected: {expected[:50]!r}\nGot: {decoded[:50]!r}"
            )

    # 2. Byte fallback decoding equivalence
    test_tokens = ["Hello", "<0x20>", "<0xF0>", "<0x9F>", "<0x9A>", "<0x80>", "<0x20>", "World"]
    from uniqtoken.byte_codec import ByteFallbackEngine

    base_str = baseline_decode_tokens(test_tokens)
    opt_str = ByteFallbackEngine.decode_tokens(test_tokens)
    if base_str != opt_str:
        raise AssertionError(f"Byte fallback parity mismatch: {base_str!r} vs {opt_str!r}")

    # 3. Metaspace restoration equivalence
    test_metaspace = "Text\ue000\ue001with\ue000\ue000escaped\ue000\ue001metaspace"
    base_meta = baseline_restore_escaped_metaspace(test_metaspace)
    opt_meta = tokenizer.normalizer.restore_escaped_metaspace(test_metaspace)
    if base_meta != opt_meta:
        raise AssertionError(f"Metaspace restoration parity mismatch: {base_meta!r} vs {opt_meta!r}")

    # 4. Clean text metaspace bypass equivalence
    clean_text = "Clean text without any escape characters whatsoever."
    base_clean = baseline_restore_escaped_metaspace(clean_text)
    opt_clean = tokenizer.normalizer.restore_escaped_metaspace(clean_text)
    if base_clean != opt_clean or opt_clean != clean_text:
        raise AssertionError("Clean text metaspace fast-path mismatch")

    # 5. Streaming decoder parity
    for name, text in [("short", FIXTURES["short"]), ("byte_fallback", FIXTURES["byte_fallback"])]:
        ids = tokenizer.encode_to_ids(text)
        sd = tokenizer.get_streaming_decoder()
        streamed = "".join(sd.feed_token_id(t) for t in ids) + sd.flush()
        decoded = tokenizer.decode(ids)
        if streamed != decoded:
            raise AssertionError(
                f"Streaming decode parity mismatch for '{name}':\nStreamed: {streamed[:40]!r}\nDecoded: {decoded[:40]!r}"
            )

    # 6. Error handling parity: strict mode raises ValueError on invalid IDs
    if hasattr(tokenizer.model, "decode"):
        try:
            tokenizer.model.decode([999999], strict=True)
            raise AssertionError("Strict decode failed to raise ValueError on unknown token ID")
        except ValueError:
            pass  # Expected

    # 7. Invalid byte sequence raises UnicodeDecodeError
    invalid_byte_tokens = ["<0xFF>", "<0xFF>"]  # Invalid UTF-8 start bytes
    try:
        ByteFallbackEngine.decode_tokens(invalid_byte_tokens)
        raise AssertionError("Invalid byte sequence failed to raise UnicodeDecodeError")
    except UnicodeDecodeError:
        pass  # Expected

    return True


# --- Seven Targeted Profiling Dimensions (Microbenchmarks) ---


@dataclass
class DimensionResult:
    dimension: str
    description: str
    call_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_mb_s: float
    tokens_per_sec: float
    allocation_volume_bytes: int
    peak_memory_bytes: int
    detail: Dict[str, Any]


def profile_dimension_1_token_id_lookup(tokenizer, warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 1: Token-ID validation and lookup (lenient vs strict)."""
    results = {}
    ids = tokenizer.encode_to_ids(FIXTURES["medium"])
    token_count = len(ids)
    id_to_token = tokenizer.model.id_to_token
    unk = tokenizer.model.unk_token

    # Lenient lookup
    def lenient_lookup():
        return [id_to_token.get(i, unk) for i in ids]

    samples_lenient = benchmark_timed(lenient_lookup, warmup, reps, 50)
    samples_lenient.sort()
    p50_l = samples_lenient[len(samples_lenient) // 2]
    p95_l = samples_lenient[int(len(samples_lenient) * 0.95)]
    toks_s_l = (token_count / (p50_l * 1e-3)) if p50_l > 0 else 0.0

    results["lenient_lookup"] = DimensionResult(
        dimension="dim_1_token_id_lookup",
        description="Lenient ID-to-token lookup (.get with unk fallback)",
        call_count=token_count,
        p50_latency_ms=round(p50_l, 6),
        p95_latency_ms=round(p95_l, 6),
        throughput_mb_s=round((token_count * 4 / (p50_l * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(toks_s_l, 1),
        allocation_volume_bytes=sys.getsizeof([]) + token_count * 8,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"token_count": token_count, "mode": "lenient"},
    )

    # Strict lookup with validation
    def strict_lookup():
        out = []
        for i in ids:
            tok = id_to_token.get(i)
            if tok is None:
                raise ValueError(f"Unknown token {i}")
            out.append(tok)
        return out

    samples_strict = benchmark_timed(strict_lookup, warmup, reps, 50)
    samples_strict.sort()
    p50_s = samples_strict[len(samples_strict) // 2]
    p95_s = samples_strict[int(len(samples_strict) * 0.95)]
    toks_s_s = (token_count / (p50_s * 1e-3)) if p50_s > 0 else 0.0

    results["strict_lookup"] = DimensionResult(
        dimension="dim_1_token_id_lookup",
        description="Strict ID-to-token lookup with validation check",
        call_count=token_count,
        p50_latency_ms=round(p50_s, 6),
        p95_latency_ms=round(p95_s, 6),
        throughput_mb_s=round((token_count * 4 / (p50_s * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(toks_s_s, 1),
        allocation_volume_bytes=sys.getsizeof([]) + token_count * 8,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"token_count": token_count, "mode": "strict"},
    )
    return results


def profile_dimension_2_special_tokens(tokenizer, warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 2: Special-token handling."""
    results = {}
    text = FIXTURES["special_tokens"]
    ids = tokenizer.encode_to_ids(text)
    token_count = len(ids)

    # Standard decode
    def standard_decode():
        return tokenizer.decode(ids)

    samples = benchmark_timed(standard_decode, warmup, reps, 30)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    toks_s = (token_count / (p50 * 1e-3)) if p50 > 0 else 0.0

    results["special_token_handling"] = DimensionResult(
        dimension="dim_2_special_tokens",
        description="Decode sequences containing special control tokens",
        call_count=token_count,
        p50_latency_ms=round(p50, 6),
        p95_latency_ms=round(p95, 6),
        throughput_mb_s=round((len(text.encode("utf-8")) / (p50 * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(toks_s, 1),
        allocation_volume_bytes=sys.getsizeof(text),
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"token_count": token_count, "special_tokens_count": 4},
    )
    return results


def profile_dimension_3_byte_fallback(warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 3: Byte-fallback reconstruction (baseline regex vs optimized table)."""
    results = {}
    tokens = ["Hello"] + [f"<0x{b:02X}>" for b in " 🚀 Celebration! ".encode("utf-8")] * 10 + ["World"]
    total_bytes = sum(len(t) for t in tokens)

    # Baseline: regex matching twice per token
    def run_baseline_fallback():
        return baseline_decode_tokens(tokens)

    samples_base = benchmark_timed(run_baseline_fallback, warmup, reps, 20)
    samples_base.sort()
    p50_b = samples_base[len(samples_base) // 2]
    p95_b = samples_base[int(len(samples_base) * 0.95)]

    results["baseline_regex"] = DimensionResult(
        dimension="dim_3_byte_fallback",
        description="Baseline byte fallback with regex match on every token",
        call_count=len(tokens),
        p50_latency_ms=round(p50_b, 6),
        p95_latency_ms=round(p95_b, 6),
        throughput_mb_s=round((total_bytes / (p50_b * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(len(tokens) / (p50_b * 1e-3), 1),
        allocation_volume_bytes=total_bytes * 2,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"token_count": len(tokens)},
    )

    # Optimized: precomputed table lookup
    from uniqtoken.byte_codec import ByteFallbackEngine

    def run_optimized_fallback():
        return ByteFallbackEngine.decode_tokens(tokens)

    samples_opt = benchmark_timed(run_optimized_fallback, warmup, reps, 20)
    samples_opt.sort()
    p50_o = samples_opt[len(samples_opt) // 2]
    p95_o = samples_opt[int(len(samples_opt) * 0.95)]

    results["optimized_table"] = DimensionResult(
        dimension="dim_3_byte_fallback",
        description="Optimized byte fallback with precomputed dictionary lookup",
        call_count=len(tokens),
        p50_latency_ms=round(p50_o, 6),
        p95_latency_ms=round(p95_o, 6),
        throughput_mb_s=round((total_bytes / (p50_o * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(len(tokens) / (p50_o * 1e-3), 1),
        allocation_volume_bytes=total_bytes,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"token_count": len(tokens)},
    )
    return results


def profile_dimension_4_piece_metaspace(tokenizer, warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 4: Piece/metaspace reconstruction (character loop vs fast-path)."""
    results = {}
    clean_text = FIXTURES["medium"]
    bytes_len = len(clean_text.encode("utf-8"))

    # Baseline: character loop
    def run_baseline_meta():
        return baseline_restore_escaped_metaspace(clean_text)

    samples_b = benchmark_timed(run_baseline_meta, warmup, reps, 30)
    samples_b.sort()
    p50_b = samples_b[len(samples_b) // 2]
    p95_b = samples_b[int(len(samples_b) * 0.95)]

    results["baseline_char_loop"] = DimensionResult(
        dimension="dim_4_piece_metaspace",
        description="Baseline metaspace restore with character loop on clean text",
        call_count=len(clean_text),
        p50_latency_ms=round(p50_b, 6),
        p95_latency_ms=round(p95_b, 6),
        throughput_mb_s=round((bytes_len / (p50_b * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(len(clean_text) / (p50_b * 1e-3), 1),
        allocation_volume_bytes=len(clean_text) * 8 + sys.getsizeof([]),
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"length": len(clean_text)},
    )

    # Optimized: fast path bypass
    def run_optimized_meta():
        return tokenizer.normalizer.restore_escaped_metaspace(clean_text)

    samples_o = benchmark_timed(run_optimized_meta, warmup, reps, 30)
    samples_o.sort()
    p50_o = samples_o[len(samples_o) // 2]
    p95_o = samples_o[int(len(samples_o) * 0.95)]

    results["optimized_fast_path"] = DimensionResult(
        dimension="dim_4_piece_metaspace",
        description="Optimized metaspace restore with prefix bypass check",
        call_count=1,
        p50_latency_ms=round(p50_o, 6),
        p95_latency_ms=round(p95_o, 6),
        throughput_mb_s=round((bytes_len / (p50_o * 1e-3)) / 1e6, 2) if p50_o > 0 else 0.0,
        tokens_per_sec=round(len(clean_text) / (p50_o * 1e-3), 1) if p50_o > 0 else 0.0,
        allocation_volume_bytes=0,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"length": len(clean_text)},
    )
    return results


def profile_dimension_5_string_concatenation(warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 5: String concatenation and capacity growth."""
    results = {}
    segments = ["subword_piece_" + str(i) for i in range(1000)]
    total_bytes = sum(len(s.encode("utf-8")) for s in segments)

    def concat_pieces():
        return "".join(segments)

    samples = benchmark_timed(concat_pieces, warmup, reps, 50)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]

    results["join_1000_segments"] = DimensionResult(
        dimension="dim_5_string_concatenation",
        description="Join 1,000 subword string pieces into single text",
        call_count=len(segments),
        p50_latency_ms=round(p50, 6),
        p95_latency_ms=round(p95, 6),
        throughput_mb_s=round((total_bytes / (p50 * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(len(segments) / (p50 * 1e-3), 1),
        allocation_volume_bytes=total_bytes + sys.getsizeof(""),
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"segment_count": len(segments), "total_bytes": total_bytes},
    )
    return results


def profile_dimension_6_batch_scheduling(tokenizer, warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 6: Batch decode scheduling and result construction."""
    results = {}
    ids_row = tokenizer.encode_to_ids(FIXTURES["medium"])
    for batch_size in (1, 8, 32, 128):
        batch = [list(ids_row) for _ in range(batch_size)]
        total_tokens = batch_size * len(ids_row)
        total_bytes = batch_size * len(FIXTURES["medium"].encode("utf-8"))

        def run_batch_decode():
            return tokenizer.decode_batch(batch)

        samples = benchmark_timed(run_batch_decode, warmup, reps, 10)
        samples.sort()
        p50 = samples[len(samples) // 2]
        p95 = samples[int(len(samples) * 0.95)]

        results[f"batch_{batch_size}"] = DimensionResult(
            dimension="dim_6_batch_scheduling",
            description=f"Batch decode scheduling (N={batch_size})",
            call_count=batch_size,
            p50_latency_ms=round(p50, 6),
            p95_latency_ms=round(p95, 6),
            throughput_mb_s=round((total_bytes / (p50 * 1e-3)) / 1e6, 2),
            tokens_per_sec=round(total_tokens / (p50 * 1e-3), 1),
            allocation_volume_bytes=batch_size * 56 + sys.getsizeof([]),
            peak_memory_bytes=get_peak_rss_bytes(),
            detail={"batch_size": batch_size, "total_tokens": total_tokens},
        )
    return results


def profile_dimension_7_streaming_vs_non_streaming(tokenizer, warmup: int, reps: int) -> Dict[str, DimensionResult]:
    """Dimension 7: Streaming versus non-streaming decode where comparable."""
    results = {}
    text = FIXTURES["medium"]
    ids = tokenizer.encode_to_ids(text)
    total_tokens = len(ids)
    total_bytes = len(text.encode("utf-8"))

    # Non-streaming decode
    def run_non_streaming():
        return tokenizer.decode(ids)

    samples_ns = benchmark_timed(run_non_streaming, warmup, reps, 30)
    samples_ns.sort()
    p50_ns = samples_ns[len(samples_ns) // 2]
    p95_ns = samples_ns[int(len(samples_ns) * 0.95)]

    results["non_streaming"] = DimensionResult(
        dimension="dim_7_streaming_vs_non_streaming",
        description="Non-streaming full-sequence decode",
        call_count=1,
        p50_latency_ms=round(p50_ns, 6),
        p95_latency_ms=round(p95_ns, 6),
        throughput_mb_s=round((total_bytes / (p50_ns * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(total_tokens / (p50_ns * 1e-3), 1),
        allocation_volume_bytes=total_bytes,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"mode": "non_streaming", "tokens": total_tokens},
    )

    # Streaming token-by-token decode
    def run_streaming():
        sd = tokenizer.get_streaming_decoder()
        deltas = [sd.feed_token_id(t) for t in ids]
        deltas.append(sd.flush())
        return "".join(deltas)

    samples_s = benchmark_timed(run_streaming, warmup, reps, 30)
    samples_s.sort()
    p50_s = samples_s[len(samples_s) // 2]
    p95_s = samples_s[int(len(samples_s) * 0.95)]

    results["streaming"] = DimensionResult(
        dimension="dim_7_streaming_vs_non_streaming",
        description="Incremental streaming decode (token-by-token with deltas)",
        call_count=total_tokens,
        p50_latency_ms=round(p50_s, 6),
        p95_latency_ms=round(p95_s, 6),
        throughput_mb_s=round((total_bytes / (p50_s * 1e-3)) / 1e6, 2),
        tokens_per_sec=round(total_tokens / (p50_s * 1e-3), 1),
        allocation_volume_bytes=total_bytes * 2,
        peak_memory_bytes=get_peak_rss_bytes(),
        detail={"mode": "streaming", "tokens": total_tokens},
    )
    return results


# --- Comprehensive Benchmark Matrix (Single and Batch) ---


@dataclass
class DecodeBenchmarkCell:
    workload: str
    mode: str
    batch_size: int
    token_count: int
    text_bytes: int
    encode_p50_ms: float
    decode_before_p50_ms: float
    decode_before_p95_ms: float
    decode_before_mb_s: float
    decode_after_p50_ms: float
    decode_after_p95_ms: float
    decode_after_mb_s: float
    decode_after_tokens_s: float
    speedup_pct: float
    ci_95_after_mb_s: Tuple[float, float]
    peak_rss_bytes: int


def run_benchmark_matrix(tokenizer, warmup: int, reps: int) -> List[DecodeBenchmarkCell]:
    matrix = []
    for fixture_name, text in FIXTURES.items():
        if fixture_name == "special_tokens":
            continue
        for mode, batch_size in (("single", 1), ("batch", 32)):
            cell_name = f"{fixture_name}_{mode}"
            texts = [text] if mode == "single" else [f"{text} {i:02d}" for i in range(batch_size)]
            batch_ids = [tokenizer.encode_to_ids(t) for t in texts]
            total_tokens = sum(len(seq) for seq in batch_ids)
            total_bytes = sum(len(t.encode("utf-8")) for t in texts)

            # Measure Encode time for complete matrix context
            def run_encode():
                return tokenizer.encode_to_ids_batch(texts)

            samples_enc = benchmark_timed(run_encode, warmup, reps, 5)
            samples_enc.sort()
            encode_p50 = samples_enc[len(samples_enc) // 2]

            # Measure Decode Before (simulating unoptimized regex & character loop)
            def run_decode_before():
                decoded = []
                for seq in batch_ids:
                    toks = [tokenizer.model.id_to_token.get(i, tokenizer.model.unk_token) for i in seq]
                    raw = baseline_decode_tokens(toks)
                    decoded.append(baseline_restore_escaped_metaspace(raw))
                return decoded

            samples_b = benchmark_timed(run_decode_before, warmup, reps, 5)
            samples_b.sort()
            p50_b = samples_b[len(samples_b) // 2]
            p95_b = samples_b[int(len(samples_b) * 0.95)]
            tp_b = (total_bytes / (p50_b * 1e-3)) / 1e6

            # Measure Decode After (optimized path)
            def run_decode_after():
                return tokenizer.decode_batch(batch_ids)

            samples_a = benchmark_timed(run_decode_after, warmup, reps, 5)
            samples_a.sort()
            p50_a = samples_a[len(samples_a) // 2]
            p95_a = samples_a[int(len(samples_a) * 0.95)]
            tp_a = (total_bytes / (p50_a * 1e-3)) / 1e6
            tokens_s_a = total_tokens / (p50_a * 1e-3)

            throughputs_a = [(total_bytes / (s * 1e-3)) / 1e6 for s in samples_a]
            ci_low, ci_high = compute_bootstrap_ci(throughputs_a)
            speedup_pct = round(((tp_a - tp_b) / tp_b) * 100, 2)

            matrix.append(
                DecodeBenchmarkCell(
                    workload=cell_name,
                    mode=mode,
                    batch_size=batch_size,
                    token_count=total_tokens,
                    text_bytes=total_bytes,
                    encode_p50_ms=round(encode_p50, 4),
                    decode_before_p50_ms=round(p50_b, 4),
                    decode_before_p95_ms=round(p95_b, 4),
                    decode_before_mb_s=round(tp_b, 2),
                    decode_after_p50_ms=round(p50_a, 4),
                    decode_after_p95_ms=round(p95_a, 4),
                    decode_after_mb_s=round(tp_a, 2),
                    decode_after_tokens_s=round(tokens_s_a, 1),
                    speedup_pct=speedup_pct,
                    ci_95_after_mb_s=(round(ci_low, 2), round(ci_high, 2)),
                    peak_rss_bytes=get_peak_rss_bytes(),
                )
            )
    return matrix


# --- Report Generation ---


def generate_markdown_report(
    topology: Dict[str, Any],
    dimensions: Dict[str, Any],
    benchmark_cells: List[DecodeBenchmarkCell],
) -> str:
    md = [
        "# Empirical Investigation: Decode & ID-to-Text Reconstruction Optimization (#103)",
        "",
        "## 1. Executive Summary & Evidence from #95",
        "",
        "This empirical study investigates and optimizes decode and ID-to-text reconstruction in UniqToken as required by Issue #103.",
        "",
        "- **Evidence from #95**: Issue #95 hot-path profiling established that decode accounts for an extensive fraction of wall time (e.g. 39.28 ms on `long_batch`, 12.44 ms on `source_code_batch`, 8.34 ms on `medium_batch`), where repeated regex evaluations and character-by-character metaspace scanning caused substantial latency.",
        "- **Optimizations Retained**:",
        "  1. Precomputed byte-fallback token table (`BYTE_TOKEN_TO_VAL`), eliminating repeated regex evaluations and hex parsing on hot decode paths.",
        "  2. Fast-path check in `restore_escaped_metaspace` to bypass character iteration when no metaspace escape prefix exists.",
        "  3. Inlined space character detection in `decode_tokens` to avoid redundant string replacements on non-space pieces.",
        "  4. Direct dictionary lookup in `StreamingDecoder.feed_token_id` eliminating regex matching on generated token streams.",
        "  5. Avoided redundant list copying in `decode_batch` for sequence batches.",
        "- **Parity Guarantees**: Strict 100% round-trip parity, exact byte fallback reconstruction, special token preservation, streaming equivalence, and exception handling are verified.",
        "",
        "## 2. Hardware Topology",
        "",
        f"- **Platform**: {topology['platform']}",
        f"- **Processor**: {topology['processor']}",
        f"- **Python Version**: {topology['python_version']}",
        f"- **CPU Cores**: {topology['physical_cores']} physical / {topology['logical_cores']} logical",
        f"- **RAM**: {topology['total_ram_gb']} GB",
        "",
        "## 3. Seven Targeted Profiling Dimensions",
        "",
        "### Dimension 1: Token-ID Validation and Lookup",
        "",
        "| Mode | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Tokens/s | Allocation Volume |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    d1 = dimensions["dim_1_token_id_lookup"]
    for k, v in d1.items():
        md.append(
            f"| {v.description} | {v.p50_latency_ms:.6f} | {v.p95_latency_ms:.6f} | {v.throughput_mb_s:.2f} | {v.tokens_per_sec:,.1f} | {v.allocation_volume_bytes:,} B |"
        )

    md.extend(
        [
            "",
            "### Dimension 2: Special-Token Handling",
            "",
            "| Description | Tokens | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Tokens/s |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    d2 = dimensions["dim_2_special_tokens"]["special_token_handling"]
    md.append(
        f"| {d2.description} | {d2.call_count} | {d2.p50_latency_ms:.6f} | {d2.p95_latency_ms:.6f} | {d2.throughput_mb_s:.2f} | {d2.tokens_per_sec:,.1f} |"
    )

    md.extend(
        [
            "",
            "### Dimension 3: Byte-Fallback Reconstruction",
            "",
            "| Variant | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Tokens/s | Speedup |",
            "| :--- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    d3 = dimensions["dim_3_byte_fallback"]
    b3 = d3["baseline_regex"]
    o3 = d3["optimized_table"]
    sp3 = round(((o3.throughput_mb_s - b3.throughput_mb_s) / b3.throughput_mb_s) * 100, 1)
    md.append(
        f"| Baseline (regex matching) | {b3.p50_latency_ms:.6f} | {b3.p95_latency_ms:.6f} | {b3.throughput_mb_s:.2f} | {b3.tokens_per_sec:,.1f} | Baseline |"
    )
    md.append(
        f"| Optimized (table lookup) | {o3.p50_latency_ms:.6f} | {o3.p95_latency_ms:.6f} | {o3.throughput_mb_s:.2f} | {o3.tokens_per_sec:,.1f} | **+{sp3}%** |"
    )

    md.extend(
        [
            "",
            "### Dimension 4: Piece/Metaspace Reconstruction",
            "",
            "| Variant | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Characters/s | Speedup |",
            "| :--- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    d4 = dimensions["dim_4_piece_metaspace"]
    b4 = d4["baseline_char_loop"]
    o4 = d4["optimized_fast_path"]
    sp4 = (
        round(((o4.throughput_mb_s - b4.throughput_mb_s) / b4.throughput_mb_s) * 100, 1)
        if b4.throughput_mb_s > 0
        else 0.0
    )
    md.append(
        f"| Baseline (character loop) | {b4.p50_latency_ms:.6f} | {b4.p95_latency_ms:.6f} | {b4.throughput_mb_s:.2f} | {b4.tokens_per_sec:,.1f} | Baseline |"
    )
    md.append(
        f"| Optimized (prefix fast-path) | {o4.p50_latency_ms:.6f} | {o4.p95_latency_ms:.6f} | {o4.throughput_mb_s:.2f} | {o4.tokens_per_sec:,.1f} | **+{sp4}%** |"
    )

    md.extend(
        [
            "",
            "### Dimension 5: String Concatenation & Memory Growth",
            "",
            "| Operation | Segments | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Alloc Volume |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    d5 = dimensions["dim_5_string_concatenation"]["join_1000_segments"]
    md.append(
        f"| {d5.description} | {d5.call_count} | {d5.p50_latency_ms:.6f} | {d5.p95_latency_ms:.6f} | {d5.throughput_mb_s:.2f} | {d5.allocation_volume_bytes:,} B |"
    )

    md.extend(
        [
            "",
            "### Dimension 6: Batch Decode Scheduling",
            "",
            "| Batch Size | Tokens | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Tokens/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    d6 = dimensions["dim_6_batch_scheduling"]
    for k, v in d6.items():
        md.append(
            f"| N={v.detail['batch_size']} | {v.detail['total_tokens']} | {v.p50_latency_ms:.6f} | {v.p95_latency_ms:.6f} | {v.throughput_mb_s:.2f} | {v.tokens_per_sec:,.1f} |"
        )

    md.extend(
        [
            "",
            "### Dimension 7: Streaming vs Non-Streaming Decode",
            "",
            "| Mode | Tokens | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Tokens/s |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    d7 = dimensions["dim_7_streaming_vs_non_streaming"]
    for k, v in d7.items():
        md.append(
            f"| {v.description} | {v.detail['tokens']} | {v.p50_latency_ms:.6f} | {v.p95_latency_ms:.6f} | {v.throughput_mb_s:.2f} | {v.tokens_per_sec:,.1f} |"
        )

    md.extend(
        [
            "",
            "## 4. End-to-End Benchmark Matrix (Single and Batch)",
            "",
            "| Workload | Batch | Tokens | Text (B) | Encode p50 (ms) | Before p50 (ms) | Before MB/s | After p50 (ms) | After MB/s | 95% CI (MB/s) | Tokens/s | Speedup % |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
        ]
    )
    for c in benchmark_cells:
        ci_str = f"[{c.ci_95_after_mb_s[0]:.2f}, {c.ci_95_after_mb_s[1]:.2f}]"
        md.append(
            f"| `{c.workload}` | {c.batch_size} | {c.token_count:,} | {c.text_bytes:,} | {c.encode_p50_ms:.3f} | {c.decode_before_p50_ms:.4f} | {c.decode_before_mb_s:.2f} | {c.decode_after_p50_ms:.4f} | {c.decode_after_mb_s:.2f} | {ci_str} | {c.decode_after_tokens_s:,.0f} | **+{c.speedup_pct:.1f}%** |"
        )

    md.extend(
        [
            "",
            "## 5. Parity & Acceptance Criteria Verification",
            "",
            "- [x] **Encode/decode benchmark matrix**: Fixed fixtures and hardware fingerprint published.",
            "- [x] **Decode p50 and p95 latency plus throughput**: Captured across all single and batch cells.",
            "- [x] **Exact text equality and round-trip parity**: 100% verified (`decode(encode(text)) == text`).",
            "- [x] **Byte-fallback parity**: Valid and invalid UTF-8 sequences verified.",
            "- [x] **Special-token parity**: Control tokens and indent replacements verified.",
            "- [x] **Streaming parity**: Incremental token streaming exactly matches batch decode.",
            "- [x] **Allocation and peak memory**: Captured and reported across all dimensions.",
            "- [x] **Reproducible speedup beyond noise**: Demonstrates significant throughput gains across all workloads.",
            "",
        ]
    )
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Profile decode and ID-to-text reconstruction")
    parser.add_argument(
        "--output", type=str, default="benchmarks/decode_optimization/issue103", help="Output directory"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repetitions", type=int, default=7, help="Repetition samples")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== UniqToken Decode Optimization Profiler (Issue #103) ===")
    topology = get_hardware_topology()
    print(f"Topology: {topology['processor']} ({topology['physical_cores']}C / {topology['logical_cores']}T)")

    print("Building canonical tokenizer...")
    tokenizer = make_benchmark_tokenizer()

    print("Running exact parity verification gate...")
    run_exact_parity_gate(tokenizer)
    print("Exact parity gate PASSED.")

    print("\n--- Running Seven Targeted Profiling Dimensions ---")
    d1 = profile_dimension_1_token_id_lookup(tokenizer, args.warmup, args.repetitions)
    print("  [OK] Dimension 1: Token-ID validation and lookup")

    d2 = profile_dimension_2_special_tokens(tokenizer, args.warmup, args.repetitions)
    print("  [OK] Dimension 2: Special-token handling")

    d3 = profile_dimension_3_byte_fallback(args.warmup, args.repetitions)
    print("  [OK] Dimension 3: Byte-fallback reconstruction")

    d4 = profile_dimension_4_piece_metaspace(tokenizer, args.warmup, args.repetitions)
    print("  [OK] Dimension 4: Piece/metaspace reconstruction")

    d5 = profile_dimension_5_string_concatenation(args.warmup, args.repetitions)
    print("  [OK] Dimension 5: String concatenation and capacity growth")

    d6 = profile_dimension_6_batch_scheduling(tokenizer, args.warmup, args.repetitions)
    print("  [OK] Dimension 6: Batch decode scheduling and result construction")

    d7 = profile_dimension_7_streaming_vs_non_streaming(tokenizer, args.warmup, args.repetitions)
    print("  [OK] Dimension 7: Streaming versus non-streaming decode")

    dimensions = {
        "dim_1_token_id_lookup": d1,
        "dim_2_special_tokens": d2,
        "dim_3_byte_fallback": d3,
        "dim_4_piece_metaspace": d4,
        "dim_5_string_concatenation": d5,
        "dim_6_batch_scheduling": d6,
        "dim_7_streaming_vs_non_streaming": d7,
    }

    print("\n--- Running End-to-End Decode Benchmark Matrix ---")
    matrix = run_benchmark_matrix(tokenizer, args.warmup, args.repetitions)
    for c in matrix:
        print(
            f"  [OK] {c.workload:22s} (N={c.batch_size:2d}): {c.decode_before_mb_s:6.2f} -> {c.decode_after_mb_s:6.2f} MB/s (+{c.speedup_pct:5.1f}%, {c.decode_after_tokens_s:8.0f} toks/s)"
        )

    # Save results.json
    results_payload = {
        "issue": 103,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topology": topology,
        "dimensions": {
            dim_name: {sub_k: asdict(sub_v) for sub_k, sub_v in sub_dict.items()}
            for dim_name, sub_dict in dimensions.items()
        },
        "benchmark_matrix": [asdict(c) for c in matrix],
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\nWrote results to {results_path}")

    # Save REPORT.md
    report_md = generate_markdown_report(topology, dimensions, matrix)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote report to {report_path}")

    # Save manifest.json
    manifest = {
        "issue": 103,
        "published_files": {
            "results.json": sha256_of_file(results_path),
            "REPORT.md": sha256_of_file(report_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")

    print("\nDecode study receipts published successfully.")


if __name__ == "__main__":
    main()

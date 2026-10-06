# Empirical Investigation: Python-Rust Boundary & FFI Overhead (#101)

## 1. Executive Summary & Evidence from #95

This empirical study profiles residual Python/Rust boundary and FFI overhead in UniqToken as mandated by Issue #101.

- **Evidence from #95**: Issue #95 demonstrated that materialization and boundary operations represent a meaningful time fraction (e.g. output copying took 1.35 ms on source code and 2.31 ms on long batches), but noted that the FFI boundary, allocation, and copying overlap and were treated as an opaque native call in Python call-stacks.
- **Coordination with #96 / #99**: Issue #96 and #99 (PR #131) bounded internal Viterbi scratch allocations and eliminated temporary owned prefix vectors. This report isolates the residual PyO3/boundary operations without attributing internal Rust lattice memory to PyO3.
- **Preservation of Guarantees**: Public API signatures and return types are strictly preserved (`List[List[int]]`, `List[List[str]]`). Safe zero-copy buffer borrowing across Rayon threads (`PyBackedStr`) is retained.

## 2. Hardware & Environment Topology

- **Platform**: Windows-11-10.0.26200-SP0
- **Processor**: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- **Python Version**: 3.13.1
- **Physical / Logical Cores**: 8 physical / 12 logical
- **System RAM**: 7.65 GB

## 3. Boundary Stage Decomposition

Decomposition of execution time into Input Conversion ($T_{\text{input}}$), Native Computation ($T_{\text{native}}$), and Result Materialization ($T_{\text{mat}}$):

| Workload | Input Bytes | Total (ms) | Input Conv (ms) | Input % | Native Comp (ms) | Native % | Materialization (ms) | Mat % | Throughput (MB/s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_single` | 44 | 0.0572 | 0.0004 | 0.8% | 0.0565 | 98.7% | 0.0003 | 0.5% | 0.77 |
| `short_batch` | 1,504 | 1.6432 | 0.0027 | 0.2% | 1.6382 | 99.7% | 0.0022 | 0.1% | 0.92 |
| `medium_single` | 1,020 | 0.6191 | 0.0003 | 0.1% | 0.6182 | 99.8% | 0.0007 | 0.1% | 1.65 |
| `medium_batch` | 32,736 | 21.4145 | 0.0036 | 0.0% | 21.4002 | 99.9% | 0.0107 | 0.1% | 1.53 |
| `long_single` | 4,730 | 3.9033 | 0.0004 | 0.0% | 3.9018 | 100.0% | 0.0011 | 0.0% | 1.21 |
| `long_batch` | 151,456 | 112.8020 | 0.0063 | 0.0% | 112.6971 | 99.9% | 0.0986 | 0.1% | 1.34 |
| `multilingual_single` | 792 | 10.6818 | 0.0259 | 0.2% | 10.6554 | 99.8% | 0.0005 | 0.0% | 0.07 |
| `multilingual_batch` | 25,440 | 356.8689 | 0.7337 | 0.2% | 356.1266 | 99.8% | 0.0086 | 0.0% | 0.07 |
| `source_code_single` | 1,050 | 0.9906 | 0.0003 | 0.0% | 0.9887 | 99.8% | 0.0016 | 0.2% | 1.06 |
| `source_code_batch` | 33,696 | 33.4786 | 0.0038 | 0.0% | 33.3953 | 99.8% | 0.0796 | 0.2% | 1.01 |

## 4. Targeted Boundary Crossings (Microbenchmarks)

### Crossing 1: Python String to Rust UTF-8 Access (Input Conversion)

| Variant | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Copied Bytes | Memory (RSS) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline (character scan) | 1.49460 | 1.51785 | 21.92 | 0 B | 0 B |
| Optimized (`isascii()` fast-path) | 0.00085 | 0.00086 | 38313.45 | 0 B | 0 B |

### Crossing 2: Rust Token IDs to Python Containers (Result Materialization)

| Token Count | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Copied Bytes | Alloc Volume |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 0.000128 | 0.000134 | 2000.00 | 256 B | 2,360 B |
| 512 tokens | 0.000642 | 0.000656 | 3190.03 | 2,048 B | 18,488 B |
| 4096 tokens | 0.006676 | 0.008564 | 2454.16 | 16,384 B | 147,512 B |

### Crossing 3: Batch Output Construction (Nested Containers)

| Batch Size | Total Tokens | Latency p50 (ms) | Latency p95 (ms) | Container Alloc Bytes | Memory (RSS) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| N=1 | 32 | 0.000170 | 0.000187 | 624 B | 0 B |
| N=8 | 256 | 0.000737 | 0.000743 | 4,600 B | 0 B |
| N=32 | 1024 | 0.002700 | 0.003013 | 18,232 B | 0 B |
| N=128 | 4096 | 0.010460 | 0.010730 | 72,760 B | 0 B |

### Crossing 4: Python Object Creation & Reference Overhead

| Object Type | Item Count | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) | Alloc Volume |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `PyLong` (> 256) | 1000 | 0.022807 | 0.036267 | 175.39 | 28,056 B |
| `PyUnicode` | 1000 | 0.149483 | 0.181373 | 53.52 | 49,056 B |

### Crossing 5: Repeated FFI Crossings Across Encode/Decode Variants

| Crossing Variant | Call Count | Latency p50 (ms) | Latency p95 (ms) | Throughput (MB/s) |
| :--- | ---: | ---: | ---: | ---: |
| Baseline pre-check (unconditional NFKC) | 32 | 1.53153 | 2.04716 | 21.39 |
| Optimized pre-check (`<` / `|` bypass) | 32 | 0.00375 | 0.00377 | 8735.47 |
| Iterative crossings (Python loop of N calls) | 32 | 21.73518 | 22.06518 | 1.51 |
| Batched crossing (1 fused native call) | 1 | 21.27532 | 22.73450 | 1.54 |

## 5. End-to-End Before vs After Performance Matrix

| Workload | Input Bytes | Before p50 (ms) | Before MB/s | After p50 (ms) | After MB/s | 95% CI (MB/s) | Speedup % |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| `short_single` | 44 | 0.0490 | 0.90 | 0.0472 | 0.93 | [0.74, 0.94] | +4.0% |
| `short_batch` | 1,504 | 1.7911 | 0.84 | 1.6010 | 0.94 | [0.90, 0.96] | +11.9% |
| `medium_single` | 1,020 | 0.8010 | 1.27 | 0.6366 | 1.60 | [1.51, 1.64] | +25.8% |
| `medium_batch` | 32,736 | 23.2042 | 1.41 | 21.9386 | 1.49 | [1.40, 1.53] | +5.8% |
| `long_single` | 4,730 | 3.6472 | 1.30 | 3.3455 | 1.41 | [1.15, 1.43] | +9.0% |
| `long_batch` | 151,456 | 218.5270 | 0.69 | 193.4307 | 0.78 | [0.75, 0.82] | +13.0% |
| `multilingual_single` | 792 | 17.9365 | 0.04 | 14.5212 | 0.05 | [0.04, 0.06] | +23.5% |
| `multilingual_batch` | 25,440 | 580.1827 | 0.04 | 554.2311 | 0.05 | [0.05, 0.05] | +4.7% |
| `source_code_single` | 1,050 | 2.0265 | 0.52 | 1.5633 | 0.67 | [0.56, 0.82] | +29.6% |
| `source_code_batch` | 33,696 | 58.7233 | 0.57 | 52.6559 | 0.64 | [0.61, 0.69] | +11.5% |

## 6. Verification & Exact Parity Gate

- **Parity Gate**: 100% verified exact token match, token ID match, offset span alignment, and decode round-trip across all fixtures.
- **Security Refusal**: Verified that fullwidth control token obfuscation (`＜|`) and lone surrogates are rejected with identical error semantics.
- **API Compatibility**: Zero breaking changes to public tokenizer interfaces.

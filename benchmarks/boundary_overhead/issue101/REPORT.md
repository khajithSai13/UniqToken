# Python guard optimization and boundary diagnostics (#101)

## Scope and method

This is a synthetic engineering diagnostic, not an isolated FFI-stage profile. The baseline restores the
two pre-PR Python checks inside the public API. Both variants use the same model, native binary, inputs,
and configuration; no extra precheck is added outside encode. Paired trials alternate execution order.
Memory is measured separately with tracemalloc and covers only the Python traced heap, not Rust allocations.

- Source commit: `a7594049934093d6b28ac3ff66dac26281f69b8a`; clean tracked tree: `True`.
- Platform: `Windows-10-10.0.26300-SP0`; Python `3.10.11`; Unicode `13.0.0`.
- Native SHA-256: `e56883b67e9bb085d6b9206d7c47cf98e239d483b2275282fefea2543f55d87e`; declared build mode: `release`.
- Rayon threads: 1; warmup: 3; repetitions: 11;
  calls per timed public-API trial: 5.
- Fixtures: five embedded text workloads, each single and batch of 32; normalized UTF-8 decimal MB/s.
- Intervals bootstrap the paired median ratios within one process; they do not establish cross-machine effects.
- Reproduce with a fresh release extension and committed source:
  `python -m benchmarks.profile_boundary_overhead --output <new-directory> --threads 1 --warmup 3 --repetitions 11 --build-mode release`.

## Public IDs API results

| Workload | Before MB/s | After MB/s | Before p50/p95 ms | After p50/p95 ms | Paired speed ratio [95% interval] |
| --- | ---: | ---: | --- | --- | --- |
| short_single | 0.523 | 0.789 | 0.1146/0.1591 | 0.0761/0.0926 | 1.407 [1.346, 1.620] |
| short_batch | 0.968 | 1.311 | 2.1480/2.6287 | 1.5871/2.0354 | 1.260 [1.196, 1.422] |
| medium_single | 0.464 | 0.500 | 2.7173/3.1584 | 2.5195/2.9306 | 1.087 [0.965, 1.146] |
| medium_batch | 0.546 | 0.622 | 74.1508/74.5916 | 65.1000/65.8132 | 1.143 [1.104, 1.179] |
| long_single | 0.810 | 0.990 | 7.3303/7.5950 | 5.9979/6.1484 | 1.223 [1.204, 1.252] |
| long_batch | 0.815 | 0.995 | 233.4552/242.8840 | 191.1192/198.4361 | 1.239 [1.197, 1.266] |
| multilingual_single | 0.465 | 0.447 | 2.0373/2.7070 | 2.1186/2.8106 | 1.043 [0.799, 1.193] |
| multilingual_batch | 0.375 | 0.384 | 81.2683/85.7588 | 79.4156/80.6502 | 1.027 [0.993, 1.047] |
| source_code_single | 0.377 | 0.398 | 3.4707/3.7893 | 3.2885/6.0839 | 1.107 [1.064, 1.138] |
| source_code_batch | 0.407 | 0.445 | 103.2961/105.8729 | 94.6077/97.0237 | 1.097 [1.076, 1.132] |

Ratios above 1 favor the optimized checks. All cells are reported, including regressions and uncertain
intervals. These comparisons change both checks together; they do not attribute gains to either check alone.

## Boundary diagnostics and parity

The first four probe groups are Python-only operations: surrogate scanning, shallow copies of existing
integer references, nested-list creation, and Python object creation. They do not time Rust UTF-8 borrowing
or u32-to-Python conversion. Copied native bytes and Rust allocation volume remain unmeasured.

The fifth group verifies 32 actual single native IDs calls versus one fused native batch IDs call, then
times those public operations without instrumentation. Their output IDs must match exactly.

Baseline/optimized parity passed for 29 cases covering tokens, IDs, raw offsets, batch
outputs, decode, invalid IDs, compatibility markers, private-use escapes, surrogates, and security policies.
Ordinary fixture spans are also checked against an independent normalized-text alignment oracle.

Independent public-batch and Python-check timings are diagnostic operations, not additive stages.
Native compute and materialization fields are null because they have not been isolated.

## Remaining work

Issue #101 remains partially addressed: isolated native input access, materialization, copied bytes, and
allocation-volume attribution still require native instrumentation. No universal performance claim follows
from these synthetic measurements. Frozen research artifacts and release configuration are unchanged.

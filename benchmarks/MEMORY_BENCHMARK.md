# Tokenizer Memory Benchmark Guide (Issue #104)

Reproducible memory profiling and characterization suite for tokenizer model loading, inference scratch buffers, batch memory scaling, and long-document memory behavior.

## Execution

From the repository root, run the fixed memory benchmark suite:

```powershell
python -m benchmarks.profile_memory --output benchmarks/memory/issue104
```

The one profiling command is `python -m benchmarks.profile_memory --output benchmarks/memory/issue104`. It defaults to 5 repetitions per workload cell, strictly isolated single-worker subprocesses for cold-start and memory measurements, and comprehensive long-document scaling up to 1 MB.

## Scope & Measured Dimensions

1. **Process Peak RSS (Working Set)**:
   Measures lifetime and active process Peak Working Set via `psutil` (`pmem.peak_wset` on Windows, or `pmem.rss`), capturing physical memory residency and C++/Rust native heap allocations.
2. **Allocation Count & Allocated Bytes**:
   Tracked with byte-exact precision via Python's standard `tracemalloc` instrumentation, recording live heap bytes, retained bytes, and allocation peaks.
3. **Loaded Vocabulary & Prefix Trie Memory**:
   Quantified via `CustomTokenizer.memory_footprint()`, breaking down vocabulary dictionaries, token-to-ID mappings, reverse tables, and recursive `PrefixTrie` nodes (total nodes, terminal nodes, and deep structure bytes).
4. **Temporary Encode / Decode Buffers**:
   Isolates active scratch allocation volume during execution (`peak_traced - baseline_traced`) from steady-state model baseline memory.
5. **Batch Memory Growth**:
   Evaluates batch sizes $N \in \{1, 8, 32, 128\}$ across `short`, `medium`, `long`, `multilingual`, and `source_code` inputs (aligned with issue #99 batch-throughput matrix).
6. **Long-Document Memory Behavior**:
   Evaluates document scaling from 1 KB to 1 MB ($1,024 \to 1,048,576$ UTF-8 bytes) to verify strictly bounded $O(L)$ linear memory scaling without intermediate buffer explosions.
7. **Cold-Start Separation**:
   Separates baseline Python process RSS, library import overhead ($\Delta \text{RSS}$), model file deserialization ($\Delta \text{RSS}$), and steady-state inference memory.

## Baseline Tokenizers

Trained deterministically on the exact 400-piece fixture corpus established by issue #95 (`list(FIXTURES.values()) * 3`):
- **UniqToken**: Pure Python Unigram engine (noting that `uniqtoken_core` native Rust binary was unavailable in this Windows host environment due to missing MSVC `link.exe`).
- **SentencePiece**: Prebuilt SentencePiece C++ runtime (`0.2.2`), Unigram model with byte fallback (`spm.SentencePieceProcessor`).
- **HuggingFace Tokenizers**: Prebuilt Tokenizers Rust runtime (`0.23.2`), Byte-Level BPE model (`tokenizers.Tokenizer`).

## Published Receipts

The published directory `benchmarks/memory/issue104/` contains:
- `results.json`: Full machine-readable records with explicit units, platform specifications, hardware topology, artifact hashes, and complete measurement metrics.
- `REPORT.md`: Comprehensive analytical report including cold-start table, full workload matrix, batch growth curves, long-document scaling analysis, and acceptance criteria verification.
- `manifest.json`: Cryptographic SHA-256 hashes of `results.json` and `REPORT.md`.
- `artifacts/`: Serialized model artifacts (`uniqtoken_model/`, `spm.model`, `hf_tokenizer.json`) and source corpus for bitwise reproducibility.

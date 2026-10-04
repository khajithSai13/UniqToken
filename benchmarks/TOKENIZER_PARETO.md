# Tokenizer Pareto protocol

Issue #93 consumes the complete #92 scaling receipt, verifies every published
file and every successful model hash, then validates the complete predeclared
15-condition matrix. It reads no corpus, trains no model and opens no held-out
test file. Export source identity is distinct from measurement source identity.

Compute separate fronts for validation aggregate, language, domain and
domain/language stratum. Every successful condition must have identical scope
coverage and normalized byte, character and document counts. Within each scope,
maximize normalized bytes/token and minimize fallback percentage, emitted token
count, total vocabulary entries and training wall seconds. Bytes/token and token
count share a fixed byte denominator and are redundant dimensions, not separate
evidence. Controls and byte leaves count toward vocabulary size.

Point A dominates B exactly when A is no worse on every named axis and strictly
better on at least one. The deterministic tie tolerance is zero; equal vectors
are explicit ties and neither dominates. No weighted score is defined. Missing
and failed conditions remain visible, unclassified in every scope. They never
receive favorable imputed values. Nonfinite and invalid axes reject the export.

Training cost has one wall-time observation per model. Predeclared hypothetical
independent +/-1% and +/-5% intervals give sensitivity scenarios, not confidence
intervals, measured uncertainty or significance claims. Robust dominance holds
only if A's worst time is no worse than B's best time, with the other measured
axes unchanged. This interval order avoids the cycles that pairwise epsilon
comparisons can introduce. JSON/CSV retain all dominators, exact ties and the
classification at each sensitivity interval for every scope.

Plots show the five-axis classification projected onto bytes/token and fallback.
Circle means non-dominated on all five axes; x means dominated. Vocabulary size
sets marker area and tokenizer family sets color. A point that looks inferior
on the two plotted axes can remain on the five-axis frontier due to vocabulary
or training cost. Per-language plots retain all observed validation languages.
Missing English/code validation cannot be inferred from training metrics.

These dimensions and this bounded diagnostic corpus do not establish a global
tokenizer winner or downstream LM quality. Frozen Phase A/B/C remain unchanged.

```powershell
python -m benchmarks.receipt_archive unpack --source benchmarks/scaling/issue92 --output artifacts/issue92-original
python -m benchmarks.tokenizer_pareto --scaling artifacts/issue92-original/results.json --output artifacts/tokenizer-pareto
python -m unittest tests.test_tokenizer_pareto -v
```

Use a clean committed checkout and a new output outside the source artifact
directory. The receipt hashes results, classifications, report and both plots.

The committed `pareto/issue93/evidence.zip` preserves the complete original
classification JSON and receipt. The report, CSV and both plots remain readable;
`archive.json` binds the archive and previews. Tests extract and verify both
studies before recomputing every classification against the original scaling
hashes. To inspect all JSON fields, extract to a new directory:

```powershell
python -m benchmarks.receipt_archive unpack --source benchmarks/pareto/issue93 --output artifacts/issue93-original
```

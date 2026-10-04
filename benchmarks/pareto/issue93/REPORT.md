# Tokenizer Pareto analysis

Every classification names one validation scope and these five measured axes: normalized bytes/token (maximize), fallback percentage, emitted token count, total vocabulary size and training wall seconds (minimize).
Bytes/token and token count are redundant within identical source bytes; they are retained as requested, not interpreted as independent evidence.
Exact floating-point equality is the deterministic tie rule: no epsilon or arbitrary scalar score. Equal vectors do not dominate each other.
The two-dimensional plots project five-dimensional fronts. A visually worse projected point can remain on the five-axis frontier due to vocabulary or training cost.
Failed conditions remain unclassified in every scope; no favorable values are imputed. Missing English/code validation is not inferred from training.

## Aggregate timing sensitivity

Timing has one observation per condition. Hypothetical independent +/-1% and +/-5% intervals are sensitivity scenarios, not confidence intervals or measured variance. Robust dominance requires the left point's worst time to be no worse than the right point's best time. All corpus metrics and vocabulary sizes remain exact.

| Timing interval | Five-axis aggregate non-dominated conditions | Unclassified |
| --- | --- | ---: |
| +/-0% | boundary_bpe-131072, boundary_bpe-16384, boundary_bpe-32768, boundary_bpe-65536, boundary_bpe-8192, sp_unigram-16384, sp_unigram-32768, sp_unigram-65536, sp_unigram-8192, uniq_superbpe_r64-16384 | 1 |
| +/-1% | boundary_bpe-131072, boundary_bpe-16384, boundary_bpe-32768, boundary_bpe-65536, boundary_bpe-8192, sp_unigram-16384, sp_unigram-32768, sp_unigram-65536, sp_unigram-8192, uniq_superbpe_r64-16384 | 1 |
| +/-5% | boundary_bpe-131072, boundary_bpe-16384, boundary_bpe-32768, boundary_bpe-65536, boundary_bpe-8192, sp_unigram-16384, sp_unigram-32768, sp_unigram-65536, sp_unigram-8192, uniq_superbpe_r64-16384 | 1 |

## Scope sensitivity

| Scope | Domain | Language | Exact front size | +/-1% size | +/-5% size |
| --- | --- | --- | ---: | ---: | ---: |
| aggregate | all | all | 10 | 10 | 10 |
| domain | flores200 | all | 10 | 10 | 10 |
| language | all | am | 7 | 7 | 7 |
| language | all | ar | 13 | 13 | 13 |
| language | all | bg | 8 | 8 | 8 |
| language | all | bn | 4 | 4 | 4 |
| language | all | fa | 9 | 9 | 9 |
| language | all | gu | 4 | 4 | 4 |
| language | all | hi | 4 | 4 | 4 |
| language | all | ja | 13 | 13 | 13 |
| language | all | kn | 4 | 4 | 4 |
| language | all | ko | 14 | 14 | 14 |
| language | all | ml | 6 | 6 | 6 |
| language | all | mr | 4 | 4 | 4 |
| language | all | ru | 8 | 8 | 8 |
| language | all | sw | 6 | 6 | 6 |
| language | all | ta | 4 | 4 | 4 |
| language | all | te | 4 | 4 | 4 |
| language | all | uk | 8 | 8 | 8 |
| language | all | ur | 8 | 8 | 8 |
| language | all | yo | 11 | 11 | 11 |
| language | all | zh | 12 | 12 | 12 |
| stratum | flores200 | am | 7 | 7 | 7 |
| stratum | flores200 | ar | 13 | 13 | 13 |
| stratum | flores200 | bg | 8 | 8 | 8 |
| stratum | flores200 | bn | 4 | 4 | 4 |
| stratum | flores200 | fa | 9 | 9 | 9 |
| stratum | flores200 | gu | 4 | 4 | 4 |
| stratum | flores200 | hi | 4 | 4 | 4 |
| stratum | flores200 | ja | 13 | 13 | 13 |
| stratum | flores200 | kn | 4 | 4 | 4 |
| stratum | flores200 | ko | 14 | 14 | 14 |
| stratum | flores200 | ml | 6 | 6 | 6 |
| stratum | flores200 | mr | 4 | 4 | 4 |
| stratum | flores200 | ru | 8 | 8 | 8 |
| stratum | flores200 | sw | 6 | 6 | 6 |
| stratum | flores200 | ta | 4 | 4 | 4 |
| stratum | flores200 | te | 4 | 4 | 4 |
| stratum | flores200 | uk | 8 | 8 | 8 |
| stratum | flores200 | ur | 8 | 8 | 8 |
| stratum | flores200 | yo | 11 | 11 | 11 |
| stratum | flores200 | zh | 12 | 12 | 12 |

The JSON/CSV retain every dominator, exact tie, and scope-specific sensitivity classification. No single global or downstream LM winner is selected.

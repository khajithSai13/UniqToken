"""Scope-specific Pareto fronts from receipted vocabulary scaling measurements."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from benchmarks import run_research_experiments as h
from benchmarks.analyze_tokenizer_failures import csv_text
from benchmarks import vocabulary_scaling as scaling

AXES = {
    "bytes_per_token": {"direction": "maximize", "unit": "normalized UTF-8 bytes/token"},
    "byte_fallback_percent": {"direction": "minimize", "unit": "percent of emitted tokens"},
    "tokens": {"direction": "minimize", "unit": "emitted tokens"},
    "actual_vocab_size": {"direction": "minimize", "unit": "total entries, controls and bytes included"},
    "training_seconds": {"direction": "minimize", "unit": "wall seconds including serialization"},
}
SENSITIVITY = (0.0, 0.01, 0.05)


def coordinates(point):
    values = []
    for field, definition in AXES.items():
        value = point.get(field)
        h.require(type(value) in (int, float) and math.isfinite(value), f"missing/nonfinite Pareto axis: {field}")
        h.require(value >= 0, f"negative Pareto axis: {field}")
        if field != "byte_fallback_percent":
            h.require(value > 0, f"empty Pareto axis: {field}")
        else:
            h.require(value <= 100, "invalid fallback percentage")
        values.append(-value if definition["direction"] == "maximize" else value)
    return tuple(values)


def dominates(left, right, timing_fraction=0.0):
    """All axes no worse and at least one better, even at interval extremes."""
    h.require(math.isfinite(timing_fraction) and 0 <= timing_fraction < 1, "invalid timing sensitivity")
    a, b = list(coordinates(left)), list(coordinates(right))
    # Only wall time has a hypothetical uncertainty interval; corpus counts are exact.
    a[-1] *= 1 + timing_fraction
    b[-1] *= 1 - timing_fraction
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def classify(points, timing_fraction=0.0):
    ids = [p["condition_id"] for p in points]
    h.require(len(set(ids)) == len(ids), "duplicate Pareto condition")
    vectors = {p["condition_id"]: coordinates(p) for p in points}
    rows = []
    for point in points:
        identifier = point["condition_id"]
        dominators = sorted(
            p["condition_id"] for p in points if p is not point and dominates(p, point, timing_fraction)
        )
        ties = sorted(k for k, value in vectors.items() if k != identifier and value == vectors[identifier])
        rows.append(
            {
                **point,
                "classification": "dominated" if dominators else "non_dominated",
                "dominators": dominators,
                "exact_ties": ties,
                "timing_sensitivity_fraction": timing_fraction,
            }
        )
    return rows


def verified_scaling(path):
    return scaling.verified_bundle(path)


def analyze(payload):
    scaling.validate_results(payload)
    groups, expected_scopes, denominators = {}, None, {}
    for condition in payload["conditions"]:
        if condition["status"] != "complete":
            continue
        records = [r for r in condition["records"] if r["split"] == "validation"]
        scopes = {(r["scope"], r["domain"], r["language"]) for r in records}
        if expected_scopes is None:
            expected_scopes = scopes
        h.require(scopes == expected_scopes, "conditions have different validation scope coverage")
        for row in records:
            key = (row["scope"], row["domain"], row["language"])
            counts = (row["normalized_utf8_bytes"], row["unicode_characters"], row["documents"])
            if key in denominators:
                h.require(counts == denominators[key], "different validation inputs/normalization across conditions")
            denominators[key] = counts
            point = {
                "condition_id": f"{condition['tokenizer']}-{condition['vocab_budget']}",
                "tokenizer": condition["tokenizer"],
                "vocab_budget": condition["vocab_budget"],
                "scope": row["scope"],
                "domain": row["domain"],
                "language": row["language"],
                "status": "complete",
                **{field: row[field] for field in ("bytes_per_token", "byte_fallback_percent", "tokens")},
                "actual_vocab_size": condition["actual_vocab_size"],
                "training_seconds": condition["training_seconds"],
            }
            coordinates(point)
            groups.setdefault(key, []).append(point)
    h.require(groups, "no complete validation condition to analyze")
    classifications = []
    for key, points in sorted(groups.items(), key=lambda item: repr(item[0])):
        for fraction in SENSITIVITY:
            classifications.extend(classify(points, fraction))
            for condition in payload["conditions"]:
                if condition["status"] != "complete":
                    classifications.append(
                        {
                            "condition_id": f"{condition['tokenizer']}-{condition['vocab_budget']}",
                            "tokenizer": condition["tokenizer"],
                            "vocab_budget": condition["vocab_budget"],
                            "scope": key[0],
                            "domain": key[1],
                            "language": key[2],
                            "status": condition["status"],
                            "classification": "not_classified_missing_condition",
                            "dominators": [],
                            "exact_ties": [],
                            "timing_sensitivity_fraction": fraction,
                        }
                    )
    return classifications


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(scaling.COHORT, ("#1967a3", "#bb4c36", "#23825b")))

    def panel(axis, selected, title):
        for row in selected:
            if row["status"] != "complete":
                continue
            frontier = row["classification"] == "non_dominated"
            axis.scatter(
                row["bytes_per_token"],
                row["byte_fallback_percent"],
                s=30 + 15 * math.log2(row["actual_vocab_size"] / 8192 + 1),
                color=colors[row["tokenizer"]],
                marker="o" if frontier else "x",
            )
        axis.set(title=title, xlabel="Normalized bytes/token (maximize)", ylabel="Fallback % (minimize)")
        axis.grid(alpha=0.2)

    exact = [r for r in rows if r["timing_sensitivity_fraction"] == 0]
    fig, axis = plt.subplots(figsize=(9, 6))
    panel(
        axis, [r for r in exact if r["scope"] == "aggregate"], "Aggregate validation: five-axis Pareto classification"
    )
    fig.text(
        0.5,
        0.015,
        "Circle: five-axis frontier; x: dominated. Size: vocabulary. Blue: SPM, red: BPE, green: Uniq-r64.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "aggregate.png", dpi=160)
    plt.close(fig)
    languages = sorted({r["language"] for r in exact if r["scope"] == "language"})
    columns, lines = 4, math.ceil(len(languages) / 4)
    fig, axes = plt.subplots(lines, columns, figsize=(18, 3.8 * lines), squeeze=False)
    for axis, language in zip(axes.flat, languages):
        panel(axis, [r for r in exact if r["scope"] == "language" and r["language"] == language], language)
    for axis in list(axes.flat)[len(languages) :]:
        axis.set_visible(False)
    fig.suptitle("Validation languages: five-axis fronts projected onto compression and fallback", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output / "languages.png", dpi=140)
    plt.close(fig)


def report(payload):
    rows = payload["classifications"]
    lines = [
        "# Tokenizer Pareto analysis",
        "",
        "Every classification names one validation scope and these five measured axes: normalized bytes/token (maximize), fallback percentage, emitted token count, total vocabulary size and training wall seconds (minimize).",
        "Bytes/token and token count are redundant within identical source bytes; they are retained as requested, not interpreted as independent evidence.",
        "Exact floating-point equality is the deterministic tie rule: no epsilon or arbitrary scalar score. Equal vectors do not dominate each other.",
        "The two-dimensional plots project five-dimensional fronts. A visually worse projected point can remain on the five-axis frontier due to vocabulary or training cost.",
        "Failed conditions remain unclassified in every scope; no favorable values are imputed. Missing English/code validation is not inferred from training.",
        "",
        "## Aggregate timing sensitivity",
        "",
        "Timing has one observation per condition. Hypothetical independent +/-1% and +/-5% intervals are sensitivity scenarios, not confidence intervals or measured variance. Robust dominance requires the left point's worst time to be no worse than the right point's best time. All corpus metrics and vocabulary sizes remain exact.",
        "",
        "| Timing interval | Five-axis aggregate non-dominated conditions | Unclassified |",
        "| --- | --- | ---: |",
    ]
    for fraction in SENSITIVITY:
        selected = [r for r in rows if r["scope"] == "aggregate" and r["timing_sensitivity_fraction"] == fraction]
        front = sorted(r["condition_id"] for r in selected if r["classification"] == "non_dominated")
        missing = sum(r["status"] != "complete" for r in selected)
        lines.append(f"| +/-{fraction * 100:g}% | {', '.join(front)} | {missing} |")
    lines.extend(
        [
            "",
            "## Scope sensitivity",
            "",
            "| Scope | Domain | Language | Exact front size | +/-1% size | +/-5% size |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    keys = {(r["scope"], r["domain"], r["language"]) for r in rows}
    for key in sorted(keys, key=repr):
        sizes = [
            sum(
                (r["scope"], r["domain"], r["language"]) == key
                and r["timing_sensitivity_fraction"] == f
                and r["classification"] == "non_dominated"
                for r in rows
            )
            for f in SENSITIVITY
        ]
        lines.append("| " + " | ".join([key[0], key[1] or "all", key[2] or "all", *map(str, sizes)]) + " |")
    lines.extend(
        [
            "",
            "The JSON/CSV retain every dominator, exact tie, and scope-specific sensitivity classification. No single global or downstream LM winner is selected.",
            "",
        ]
    )
    return "\n".join(lines)


def run(path, output):
    h.require(not output.exists(), "output must be new")
    h.require(not output.resolve().is_relative_to(path.resolve().parent), "output overlaps source artifacts")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit Pareto protocol before exporting")
    source, receipt = verified_scaling(path)
    rows = analyze(source)
    payload = {
        "schema_version": 1,
        "identity": identity,
        "measurement_identity": source["identity"],
        "source_results_sha256": h.file_hash(path),
        "source_manifest_sha256": h.file_hash(path.parent / "manifest.json"),
        "source_artifacts": receipt["artifacts"],
        "assignments": source["assignments"],
        "axes": AXES,
        "tie_tolerance": 0,
        "timing_sensitivity_fractions": SENSITIVITY,
        "classifications": rows,
    }
    output.mkdir(parents=True)
    h.write_new_json(output / "results.json", payload)
    (output / "classifications.csv").write_text(csv_text(rows), encoding="utf-8")
    plot(rows, output)
    (output / "REPORT.md").write_text(report(payload), encoding="utf-8")
    h.require(h.runtime_identity() == identity, "source/runtime changed during export")
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.scaling, args.output)


if __name__ == "__main__":
    main()

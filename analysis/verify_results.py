"""Audit archived class totals, check benchmark blocks and rebuild the score tables."""
import argparse
import csv
import io
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = ("ResNet18", "MobileNetV2", "EfficientNetV2S")
CONTROLS = ("ReLU", "LeakyReLU", "PReLU", "ELU", "PELU", "Swish", "Mish", "PowLU")
SQUAREPLUS = ("Squareplus-fixed", "Squareplus-learned")
ABLATION = ("ZABRA", "ZABRA-Odd", "ZABRA-Even", "ReLU")
METRICS = ("top1", "macro_f1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(value, expected, context):
    require(math.isclose(float(value), float(expected), abs_tol=1e-12, rel_tol=1e-12), context)


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def table(name, rows, write):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path = RESULTS / name
    if write:
        path.write_bytes(stream.getvalue().encode("utf-8"))
    else:
        with path.open(encoding="utf-8", newline="") as existing:
            stored = list(csv.DictReader(existing))
        stream.seek(0)
        derived = list(csv.DictReader(stream))
        require(len(stored) == len(derived), name)
        for left, right in zip(stored, derived):
            require(left.keys() == right.keys(), name)
            for column, expected in right.items():
                if left[column] != expected:
                    same(left[column], expected, f"{name} {column}")


def scores_from_counts(row):
    actual, predicted, tp = (row[field] for field in ("actual", "predicted", "true_positives"))
    for counts in (actual, predicted, tp):
        require(all(type(n) is int and n >= 0 for n in counts), row["run_id"])
    require(len(actual) == len(predicted) == len(tp), row["run_id"])
    require(sum(actual) == sum(predicted) and sum(actual) > 0, row["run_id"])
    f1 = []
    for a, p, hit in zip(actual, predicted, tp):
        require(hit <= min(a, p), row["run_id"])
        f1.append(2 * hit / (a + p) if a + p else 0.)
    return sum(tp) / sum(actual), statistics.mean(f1)


def audit_runs(protocol):
    scores = read_csv("run_scores.csv")
    with (RESULTS / "class_counts.jsonl").open(encoding="utf-8") as file:
        counts = [json.loads(line) for line in file]
    by_id = {r["run_id"]: r for r in counts}
    require(len(by_id) == len(counts) == len(scores), "Score/count row cardinality")
    require({r["run_id"] for r in scores} == by_id.keys(), "Score/count identities")
    groups = defaultdict(dict)
    for row in scores:
        dataset, model, arm = row["dataset"], row["model"], row["activation"]
        seed = int(row["seed"])
        spec = protocol["datasets"][dataset]
        key = f"{dataset}.{model}.{arm}.{seed}"
        require(key == row["run_id"] and seed in protocol["seeds"] and model in MODELS, key)
        count = by_id[key]
        require(len(count["actual"]) == spec["classes"], key)
        require(sum(count["actual"]) == int(row["test_count"]) == spec["test_count"], key)
        for metric, value in zip(METRICS, scores_from_counts(count)):
            same(value, row[metric], key + " " + metric)
        require(seed not in groups[dataset, model, arm], key)
        groups[dataset, model, arm][seed] = row
    require(Counter(r["activation"] for r in scores) == protocol["training_runs"], "Arm coverage")
    return groups


def paired_rows(groups, dataset, model, arm, ref, seeds):
    left, right = groups[dataset, model, arm], groups[dataset, model, ref]
    rows = []
    for seed in seeds:
        if seed not in left or seed not in right:
            continue
        a, b = left[seed], right[seed]
        row = dict(dataset=dataset, model=model, candidate=arm, reference=ref,
                   seed=seed, candidate_id=a["run_id"], reference_id=b["run_id"])
        for metric in METRICS:
            row[metric + "_delta"] = float(a[metric]) - float(b[metric])
        rows.append(row)
    return rows


def paired_summary(dataset, model, arm, ref, rows, seeds):
    out = dict(dataset=dataset, model=model, candidate=arm, reference=ref,
               n=len(rows), planned_n=len(seeds), status="complete" if len(rows) == len(seeds) else "incomplete")
    for metric in METRICS:
        values = [r[metric + "_delta"] for r in rows]
        out[metric + "_delta_mean"] = statistics.mean(values) if values else ""
        out[metric + "_delta_sample_sd"] = statistics.stdev(values) if len(values) >= 2 else ""
    return out


def analyze(protocol, groups, write):
    seeds = protocol["seeds"]
    expected = set(seeds)
    failures = read_csv("unscored_controls.csv")
    failed_cells = {(r["dataset"], r["model"], r["activation"], int(r["seed"])) for r in failures}
    require(len(failed_cells) == len(failures) == protocol["numerical_failure_controls"], "Failure coverage")
    missing = set()
    for dataset, model in itertools.product(protocol["datasets"], MODELS):
        require(set(groups[dataset, model, "ZABRA"]) == expected, "Missing proposed model")
        for arm in CONTROLS:
            for seed in expected - groups[dataset, model, arm].keys():
                missing.add((dataset, model, arm, seed))
    require(failed_cells == missing, "Mismatch between unscored controls and missing test scores")
    summaries = []
    for (dataset, model, arm), by_seed in sorted(groups.items()):
        if not by_seed:
            continue
        out = dict(dataset=dataset, model=model, activation=arm, n=len(by_seed))
        for metric in METRICS:
            values = [float(by_seed[s][metric]) for s in seeds if s in by_seed]
            out[metric + "_mean"] = statistics.mean(values)
            out[metric + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summaries.append(out)
    table("score_summary.csv", summaries, write)
    for panel, arms in (("controls", CONTROLS), ("squareplus", SQUAREPLUS)):
        pairs, summaries = [], []
        for dataset, model in itertools.product(sorted(protocol["datasets"]), MODELS):
            if panel == "squareplus" and dataset == "imagenet100":
                continue
            for ref in arms:
                rows = paired_rows(groups, dataset, model, "ZABRA", ref, seeds)
                if panel == "squareplus":
                    require(len(rows) == len(seeds), "Incomplete Squareplus group")
                pairs.extend(rows)
                summaries.append(paired_summary(dataset, model, "ZABRA", ref, rows, seeds))
        table(panel + "_seed_pairs.csv", pairs, write)
        table(panel + "_paired_summary.csv", summaries, write)

    extracted = [groups["cifar100", "ResNet18", arm][s] for arm, s in itertools.product(ABLATION, seeds)]
    subset = read_csv("ablation_scores.csv")
    require(len(subset) == len(extracted) and len({r["run_id"] for r in subset}) == len(extracted), "Ablation rows")
    sources = {r["run_id"]: r for r in extracted}
    for row in subset:
        require(all(sources[row["run_id"]][k] == value for k, value in row.items()), "Ablation source")
    contrasts = []
    for arm, ref in itertools.combinations(ABLATION, 2):
        rows = paired_rows(groups, "cifar100", "ResNet18", arm, ref, seeds)
        require(len(rows) == len(seeds), "Incomplete ablation contrast")
        contrasts.append(paired_summary("cifar100", "ResNet18", arm, ref, rows, seeds))
    table("ablation_paired_summary.csv", contrasts, write)


def audit_benchmarks(protocol):
    meta = protocol["benchmark"]
    cells = read_csv("benchmark_cells.csv")
    blocks = defaultdict(dict)
    for row in read_csv("benchmark_blocks.csv"):
        group = blocks[row["cell"]]
        index = int(row["block"])
        value = float(row["ms_per_call"])
        require(index not in group and value > 0 and math.isfinite(value), row["cell"])
        group[index] = value
    accepted = {r["cell"] for r in cells if r["status"] == "accepted"}
    require(set(blocks) == accepted and len(accepted) == meta["measured_cells"], "Measured cells")
    require(len(cells) == len(accepted) + meta["excluded_cells"], "Exclusion count")
    expected = set(itertools.product(("ReLU", "ZABRA"), ("eager", "cudagraphs"),
                                     ("model_b1", "model_b128", "stem_forward", "stem_backward")))
    require({(r["activation"], r["backend"], r["workload"]) for r in cells} == expected, "Benchmark coverage")
    for row in cells:
        name = row["cell"]
        if name not in accepted:
            require(row["status"] == "excluded" and row["exclusion_reason"], name)
            continue
        n = int(row["blocks"])
        require(set(blocks[name]) == set(range(1, n + 1)) and n == meta["blocks"], name)
        values = list(blocks[name].values())
        q1, median, q3 = statistics.quantiles(values, method="inclusive")
        for metric, value in (("q1_ms", q1), ("median_ms", median), ("q3_ms", q3),
                              ("sample_sd_ms", statistics.stdev(values))):
            same(value, row[metric], name)
        require(int(row["peak_allocated_bytes"]) - int(row["allocated_after_warmup_bytes"])
                == int(row["peak_increment_over_warmup_bytes"]), name)
        require(int(row["peak_reserved_bytes"]) >= int(row["peak_allocated_bytes"]), name)
        require(row["numerical_check"] == "passed", name)
        if row["workload"].startswith("model_"):
            batch = int(row["shape"].split("x")[0])
            same(batch * 1000 / median, row["images_per_second"], name)
        if row["backend"] == "cudagraphs":
            require(int(row["graph_launches_before"]) > 0 and int(row["graph_launches_after"]) > 0, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the summary and seed-pair CSVs")
    args = parser.parse_args()
    protocol = json.loads((RESULTS / "protocol.json").read_bytes())
    groups = audit_runs(protocol)
    analyze(protocol, groups, args.write)
    audit_benchmarks(protocol)
    print(f"Verified {sum(protocol['training_runs'].values())} scored runs and "
          f"{protocol['benchmark']['measured_cells']} benchmark cells.")


if __name__ == "__main__":
    main()

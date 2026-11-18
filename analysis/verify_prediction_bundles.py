import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


METRICS = ("top1", "macro_f1", "macro_roc_auc_ovr")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decode_predictions(raw, expected, protocol):
    require(sha256(raw) == expected["source_predictions_sha256"], "Original prediction SHA-256 mismatch")
    with ZipFile(io.BytesIO(raw)) as nested:
        require(not nested.comment and all(not i.comment and not i.extra for i in nested.infolist()),
                "Unexpected archive metadata")
        require(set(nested.namelist()) == {"y_true.npy", "probabilities.npy", "logits.npy"},
                "Unexpected prediction archive members")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        labels, probabilities, logits = (archive[name] for name in ("y_true", "probabilities", "logits"))
    shape = (protocol["test_count"], protocol["classes"])
    require(labels.shape == (shape[0],) and labels.dtype.kind in "iu", "Invalid label shape or dtype")
    require(probabilities.shape == logits.shape == shape and probabilities.dtype.kind == logits.dtype.kind == "f",
            "Invalid floating-point tensor shape or dtype")
    require(np.isfinite(probabilities).all() and np.isfinite(logits).all(), "Non-finite probabilities or logits")
    require(np.all((probabilities >= 0) & (probabilities <= 1)), "Probability bounds")
    require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0), "Probability sums")
    labels = labels.astype("<i8")
    require(sha256(labels.tobytes()) == protocol["test_labels_sha256"], "Ordered test label SHA-256 mismatch")
    require(np.all((labels >= 0) & (labels < shape[1])), "Class labels outside the expected range")
    return labels, probabilities


def recompute_metrics(labels, probabilities):
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    matrix = np.bincount(labels * classes + predicted, minlength=classes**2).reshape(classes, classes)
    supports = matrix.sum(axis=1)
    f1 = np.mean(2.0 * matrix.diagonal() / (supports + matrix.sum(axis=0)))
    areas = []
    for label, values in enumerate(probabilities.T):
        _, groups, counts = np.unique(values, return_inverse=True, return_counts=True)
        average_ranks = counts.cumsum() - 0.5 * (counts - 1)
        positives = int(supports[label])
        require(0 < positives < len(labels), "Undefined one-vs-rest AUC")
        rank_sum = average_ranks[groups[labels == label]].sum()
        areas.append((rank_sum - positives * (positives + 1) / 2) / (positives * (len(labels) - positives)))
    return {
        "top1": float(matrix.trace() / len(labels)),
        "macro_f1": float(f1),
        "macro_roc_auc_ovr": float(np.mean(areas)),
    }


def verify_scores(observed, expected):
    for name in METRICS:
        require(abs(observed[name] - float(expected[name])) < 1e-12, f"{expected['run_id']}: {name} mismatch")


def main():
    parser = argparse.ArgumentParser(description="Recompute scores from all released prediction arrays without training.")
    parser.add_argument("--bundle-dir", required=True, type=Path, help="Directory containing four dataset ZIP files")
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, help="Optional destination for the recomputed run-level CSV")
    args = parser.parse_args()
    manifest = read_csv((args.results / "prediction_manifest.csv").read_bytes())
    bundles = read_csv((args.results / "prediction_bundles.csv").read_bytes())
    protocol = json.loads((args.results / "prediction_protocol.json").read_bytes())
    require(len(manifest) == 685 and len({row["run_id"] for row in manifest}) == 685, "Expected 685 unique scored runs")
    require(len(bundles) == 4, "Expected four datasets")
    recomputed = []
    for bundle in bundles:
        path = args.bundle_dir / bundle["file"]
        with path.open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == bundle["sha256"], "Bundle SHA-256 mismatch")
        selected = [row for row in manifest if row["dataset"] == bundle["dataset"]]
        with ZipFile(path) as archive:
            require(read_csv(archive.read("manifest.csv")) == selected, "Embedded manifest mismatch")
            sums = {name: value for value, name in
                    (line.split("  ", 1) for line in archive.read("SHA256SUMS.txt").decode().splitlines())}
            require(set(sums) | {"SHA256SUMS.txt"} == set(archive.namelist()), "Unexpected bundle members")
            require(sha256(archive.read("manifest.csv")) == sums["manifest.csv"], "Manifest checksum mismatch")
            require(sha256(archive.read("README.md")) == sums["README.md"], "README checksum mismatch")
            for row in selected:
                raw = archive.read(row["member"])
                require(sha256(raw) == sums[row["member"]], "Member checksum mismatch")
                labels, probabilities = decode_predictions(raw, row, protocol["datasets"][row["dataset"]])
                observed = recompute_metrics(labels, probabilities)
                verify_scores(observed, row)
                recomputed.append({**{k: row[k] for k in ("run_id", "dataset", "model", "activation", "seed")}, **observed})
        require(len(selected) == int(bundle["runs"]), "Run count mismatch")
        print(f"{bundle['dataset']}: {len(selected)} original prediction files verified", flush=True)
    require(len(recomputed) == 685, "Incomplete reconstruction")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output, recomputed)
    print("685 top-1, macro-F1 and macro one-vs-rest ROC-AUC values reproduced.")


if __name__ == "__main__":
    main()

"""Verify one original prediction archive against its public per-run record."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_results import RESULTS, require, same, scores_from_counts


def hash_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="identifier in results/run_scores.csv")
    parser.add_argument("predictions", type=Path, help="original test_predictions.npz")
    parser.add_argument("--result", type=Path, help="optional original run result JSON")
    args = parser.parse_args()
    with (RESULTS / "run_scores.csv").open(encoding="utf-8", newline="") as file:
        found = [r for r in csv.DictReader(file) if r["run_id"] == args.run_id]
    require(len(found) == 1, "Unknown or ambiguous run")
    row = found[0]
    protocol = json.loads((RESULTS / "protocol.json").read_bytes())
    dataset = protocol["datasets"][row["dataset"]]
    require(hash_file(args.predictions) == row["source_predictions_sha256"], "Prediction archive digest mismatch")
    if args.result:
        require(hash_file(args.result) == row["source_result_sha256"], "Run result digest mismatch")

    with np.load(args.predictions, allow_pickle=False) as source:
        truth = source["y_true"]
        probabilities = source["probabilities"]
        classes = dataset["classes"]
        require(truth.shape == (dataset["test_count"],) and truth.dtype.kind in "iu", "Test label shape or dtype")
        require(hashlib.sha256(truth.astype("<i8").tobytes()).hexdigest() == dataset["test_labels_sha256"],
                "Test label order mismatch")
        require(probabilities.shape == (len(truth), classes), "Probability shape")
        require(np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1)),
                "Invalid probability")
        require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0), "Invalid probability sums")
        indices = truth.astype(np.int64) * classes + probabilities.argmax(axis=1)
        cm = np.bincount(indices, minlength=classes**2).reshape(classes, classes)
    counts = dict(run_id=args.run_id, actual=cm.sum(axis=1).tolist(), predicted=cm.sum(axis=0).tolist(),
                  true_positives=cm.diagonal().tolist())
    with (RESULTS / "class_counts.jsonl").open(encoding="utf-8") as file:
        archived = next(r for r in map(json.loads, file) if r["run_id"] == args.run_id)
    require(counts == archived, "Class aggregates differ")
    top1, macro_f1 = scores_from_counts(counts)
    same(top1, row["top1"], "Top-1 accuracy")
    same(macro_f1, row["macro_f1"], "Macro-F1")
    print(f"{args.run_id} verified. Top-1 {top1:.6f}, macro-F1 {macro_f1:.6f}.")


if __name__ == "__main__":
    main()

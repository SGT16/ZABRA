import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def export(scores, source_index, output):
    runs = {row["run_id"]: row for row in read_csv(scores)}
    indexed = read_csv(source_index)
    require(len(indexed) == len(runs), "Source index must contain one row per scored run")
    sources = {row["run_id"]: Path(row["result_path"]) for row in indexed}
    require(set(sources) == set(runs), "Source index differs from scored runs")
    histories = []
    stems = []
    scalars = []
    coefficients = []
    provenance = []
    for run_id in sorted(runs):
        score = runs[run_id]
        result_path = sources[run_id]
        result_bytes = result_path.read_bytes()
        result_hash = digest(result_bytes)
        require(result_hash == score["source_result_sha256"], f"Result hash differs for {run_id}")
        result = json.loads(result_bytes)
        activation = result["activation"]
        if activation.endswith("-Base-Squareplus-Learned"):
            activation = "Squareplus-learned"
        elif activation.endswith("-Base-Squareplus"):
            activation = "Squareplus-fixed"
        source_identity = ".".join(str(value) for value in (result["dataset"], result["model"], activation, result["seed"]))
        require(run_id == source_identity, f"Run identity differs for {run_id}")
        identity = {key: score[key] for key in ("run_id", "dataset", "model", "activation", "seed")}
        folder = result_path.parent
        history_bytes = (folder / "history.json").read_bytes()
        gradient_bytes = (folder / "gradient_norms.csv").read_bytes()
        source_hashes = {"history.json": digest(history_bytes), "gradient_norms.csv": digest(gradient_bytes)}
        receipt_path = folder / "receipt.json"
        receipt_hash = ""
        receipt_checked = []
        if receipt_path.exists():
            receipt_bytes = receipt_path.read_bytes()
            receipt_hash = digest(receipt_bytes)
            receipt = {entry["file"]: entry["sha256"] for entry in json.loads(receipt_bytes)["files"]}
            for name, value in source_hashes.items():
                require(receipt.get(name) == value, f"Receipt differs for {run_id}/{name}")
                receipt_checked.append(name)
        history = json.loads(history_bytes)
        epochs = int(result["epochs_run"])
        require([row["epoch"] for row in history] == list(range(1, epochs + 1)), f"History epoch sequence differs for {run_id}")
        require(sum(row["amp_skipped_steps"] for row in history) == result["amp_skipped_steps_total"], f"AMP skipped-step count differs for {run_id}")
        for row in history:
            histories.append({**identity, **{key: row.get(key) for key in ("epoch", "train_loss", "val_f1", "acc", "loss", "time_sec", "learning_rate", "amp_scale_end", "amp_skipped_steps")}})
        gradients = csv.DictReader(io.StringIO(gradient_bytes.decode("utf-8-sig")))
        stem_name = "conv1.weight" if score["model"] == "ResNet18" else "features.0.0.weight"
        stem_epochs = []
        scalar_values = defaultdict(list)
        scalar_counts = defaultdict(int)
        for row in gradients:
            selected_stem = row["parameter"] == stem_name
            selected_scalar = score["activation"].startswith("ZABRA") and row["parameter"].endswith((".a", ".c"))
            if not (selected_stem or selected_scalar):
                continue
            require(row["gradient_scaling"] == "unscaled" and row["sampling"] == "first_training_minibatch_per_epoch" and row["batch_index"] == "0", f"Gradient sampling differs for {run_id}")
            epoch = int(row["epoch"])
            require(1 <= epoch <= epochs, f"Gradient epoch differs for {run_id}")
            finite = row["gradient_finite"].lower() == "true"
            gradient = float(row["gradient_l2"]) if row["gradient_l2"] else math.nan
            require(finite == math.isfinite(gradient), f"Gradient finiteness differs for {run_id}")
            if selected_stem:
                stem_epochs.append(epoch)
                stems.append({**identity, **{key: row[key] for key in ("epoch", "batch_index", "parameter", "numel", "gradient_l2", "gradient_abs_max", "parameter_l2", "gradient_finite", "amp_scale_before", "gradient_scaling", "sampling")}})
            if selected_scalar:
                require(row["numel"] == "1", f"Non-scalar coefficient gradient for {run_id}")
                key = (epoch, row["parameter"][-1])
                scalar_counts[key] += 1
                if finite:
                    scalar_values[key].append(gradient)
        require(stem_epochs == list(range(1, epochs + 1)), f"Missing or repeated stem gradient for {run_id}")
        coefficient_hash = ""
        if score["activation"].startswith("ZABRA"):
            coefficient_bytes = (folder / "adaptive_activation_parameters.csv").read_bytes()
            coefficient_hash = digest(coefficient_bytes)
            if receipt_path.exists():
                require(receipt.get("adaptive_activation_parameters.csv") == coefficient_hash, f"Coefficient receipt differs for {run_id}")
                receipt_checked.append("adaptive_activation_parameters.csv")
            sites = list(csv.DictReader(io.StringIO(coefficient_bytes.decode("utf-8-sig"))))
            expected_sites = {"ResNet18": 17, "MobileNetV2": 35, "EfficientNetV2S": 102}[score["model"]]
            require(len(sites) == expected_sites and len({row["layer"] for row in sites}) == expected_sites, f"Coefficient sites differ for {run_id}")
            terms = {"ZABRA": "ac", "ZABRA-Odd": "a", "ZABRA-Even": "c"}[score["activation"]]
            require(set(scalar_counts) == {(epoch, term) for epoch in range(1, epochs + 1) for term in terms}, f"Scalar gradient epochs or terms differ for {run_id}")
            for (epoch, term), count in sorted(scalar_counts.items()):
                require(count == expected_sites, f"Scalar gradient site count differs for {run_id}")
                values = scalar_values[epoch, term]
                scalars.append({**identity, "epoch": epoch, "term": term, "expected_sites": count, "finite_sites": len(values), "nonfinite_sites": count - len(values), "minimum_abs_gradient": min(values) if values else "", "median_abs_gradient": statistics.median(values) if values else "", "maximum_abs_gradient": max(values) if values else ""})
            for row in sites:
                require(row["activation"] == score["activation"], f"Coefficient activation differs for {run_id}")
                for raw, effective in (("a", "odd_coefficient"), ("c", "even_coefficient")):
                    require(abs(math.tanh(float(row[raw])) - float(row[effective])) < 1e-7, f"Coefficient transform differs for {run_id}")
                coefficients.append({**identity, "site": row["layer"], **{key: row[key] for key in ("a", "c", "odd_coefficient", "even_coefficient")}})
        provenance.append({**identity, "epochs": epochs, "source_result_sha256": result_hash, "source_history_sha256": source_hashes["history.json"], "source_gradient_norms_sha256": source_hashes["gradient_norms.csv"], "source_adaptive_parameters_sha256": coefficient_hash, "source_receipt_sha256": receipt_hash, "receipt_checked_files": "|".join(receipt_checked)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("training_history.csv", histories), ("stem_gradients.csv", stems), ("coefficient_gradient_epochs.csv", scalars), ("final_coefficients_verified.csv", coefficients), ("training_source_hashes.csv", provenance)):
        write_csv(output / name, rows)
        print(f"{name}: {len(rows)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.scores, arguments.source_index, arguments.output)

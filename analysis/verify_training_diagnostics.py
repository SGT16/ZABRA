import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


IDENTITY = ("run_id", "dataset", "model", "activation", "seed")
FILES = ("training_history.csv", "stem_gradients.csv", "coefficient_gradient_epochs.csv", "final_coefficients_verified.csv", "training_source_hashes.csv", "additional_source_audit.csv")
SITES = {"ResNet18": 17, "MobileNetV2": 35, "EfficientNetV2S": 102}
TERMS = {"ZABRA": "ac", "ZABRA-Odd": "a", "ZABRA-Even": "c"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows, fields, label):
    result = {tuple(str(row[field]) for field in fields): row for row in rows}
    require(len(result) == len(rows), f"Duplicate {label}")
    return result


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), label)


def valid_hash(value, label):
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"Invalid SHA-256 for {label}")


def same_identity(row, reference):
    require(all(str(row[key]) == str(reference[key]) for key in IDENTITY), f"Identity differs for {row['run_id']}")


def verify(directory, scores_path, manifest_path, configs_path):
    file_manifest = json.loads((directory / "training_files.json").read_bytes())
    require(set(file_manifest["file_sha256"]) == set(FILES), "Training file manifest coverage differs")
    for name in FILES:
        require(digest(directory / name) == file_manifest["file_sha256"][name], f"File hash differs for {name}")
    runs = {key[0]: row for key, row in indexed(read_csv(manifest_path), ("run_id",), "prediction manifest run").items()}
    scores = {key[0]: row for key, row in indexed(read_csv(scores_path), ("run_id",), "score run").items()}
    config_rows = [json.loads(line) for line in configs_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    configs = {key[0]: row for key, row in indexed(config_rows, ("run_id",), "run configuration").items()}
    require(set(scores) <= set(runs), "Original score runs are absent from the prediction manifest")
    require(set(configs) == set(runs), "Run configurations differ from prediction manifest coverage")
    for run_id, row in scores.items():
        same_identity(row, runs[run_id])
        for name in ("source_result_sha256", "source_predictions_sha256", "test_count"):
            require(row[name] == runs[run_id][name], f"Score provenance differs for {run_id}/{name}")
        for name in ("top1", "macro_f1"):
            close(row[name], runs[run_id][name], f"Manifest score differs for {run_id}/{name}")
    tables = {name: read_csv(directory / name) for name in FILES}
    history = tables["training_history.csv"]
    stems = tables["stem_gradients.csv"]
    scalars = tables["coefficient_gradient_epochs.csv"]
    coefficients = tables["final_coefficients_verified.csv"]
    source_rows = tables["training_source_hashes.csv"]
    sources = {key[0]: row for key, row in indexed(source_rows, ("run_id",), "training provenance").items()}
    require(set(sources) == set(runs), "Training provenance coverage differs")
    for run_id, source in sources.items():
        same_identity(source, runs[run_id])
        config = configs[run_id]
        same_identity(config, runs[run_id])
        require(source["source_result_sha256"] == config["source_result_sha256"] == runs[run_id]["source_result_sha256"], f"Result hash binding differs for {run_id}")
        for name in ("source_result_sha256", "source_history_sha256", "source_gradient_norms_sha256"):
            valid_hash(source[name], f"{run_id}/{name}")
        for name in ("source_receipt_sha256", "source_adaptive_parameters_sha256"):
            if source[name]:
                valid_hash(source[name], f"{run_id}/{name}")
        for name in ("source_config_sha256", "source_experiment_identity_sha256", "source_code_sha256"):
            valid_hash(config[name], f"{run_id}/{name}")
        require(config["schema_version"] == 1, f"Unsupported config schema for {run_id}")
        scientific = config["scientific_config"]
        recorded = config["recorded_settings"]
        require(int(source["epochs"]) == scientific["epochs"] == recorded["epochs_run"], f"Epoch budget differs for {run_id}")
        for plural, single in (("datasets", "dataset"), ("models", "model"), ("activations", "activation")):
            require(scientific[plural] == [config[single]], f"Config selection differs for {run_id}/{plural}")
        require(scientific["seeds"] == [int(config["seed"])], f"Config seed differs for {run_id}")
        require(recorded["test_count"] == int(runs[run_id]["test_count"]), f"Test count differs for {run_id}")
        require(bool(source["source_adaptive_parameters_sha256"]) == (source["activation"] in TERMS), f"Coefficient provenance differs for {run_id}")
        checks = set(filter(None, source["receipt_checked_files"].split("|")))
        require(checks <= {"history.json", "gradient_norms.csv", "adaptive_activation_parameters.csv"}, f"Unknown receipt check for {run_id}")
        require(not checks or bool(source["source_receipt_sha256"]), f"Receipt hash missing for {run_id}")
    history_index = indexed(history, ("run_id", "epoch"), "training epoch")
    stem_index = indexed(stems, ("run_id", "epoch"), "stem gradient epoch")
    expected_epochs = {(run_id, str(epoch)) for run_id, source in sources.items() for epoch in range(1, int(source["epochs"]) + 1)}
    require(set(history_index) == set(stem_index) == expected_epochs, "Training or stem-gradient epoch coverage differs")
    skipped = defaultdict(int)
    nonfinite_stems = 0
    for key, row in history_index.items():
        run_id, epoch = key
        same_identity(row, runs[run_id])
        for name in ("train_loss", "time_sec", "learning_rate", "amp_scale_end"):
            value = float(row[name])
            require(math.isfinite(value) and value >= 0, f"Invalid {name} for {run_id}/{epoch}")
        skipped[run_id] += int(row["amp_skipped_steps"])
        require(int(row["amp_skipped_steps"]) >= 0, f"Negative skipped updates for {run_id}/{epoch}")
        recorded = configs[run_id]["recorded_settings"]
        if recorded["validation_count"] == 0:
            require(not any(row[name] for name in ("val_f1", "acc", "loss")), f"Unexpected validation values for {run_id}/{epoch}")
        gradient = stem_index[key]
        same_identity(gradient, runs[run_id])
        expected_parameter = "conv1.weight" if row["model"] == "ResNet18" else "features.0.0.weight"
        require(gradient["parameter"] == expected_parameter, f"Unexpected gradient parameter for {run_id}/{epoch}")
        require(gradient["batch_index"] == "0" and gradient["gradient_scaling"] == "unscaled" and gradient["sampling"] == "first_training_minibatch_per_epoch", f"Gradient sampling differs for {run_id}/{epoch}")
        require(int(gradient["numel"]) > 0, f"Invalid stem parameter size for {run_id}")
        flag = gradient["gradient_finite"].lower()
        require(flag in ("true", "false"), f"Invalid gradient finite flag for {run_id}/{epoch}")
        l2 = float(gradient["gradient_l2"]) if gradient["gradient_l2"] else math.nan
        require((flag == "true") == math.isfinite(l2), f"Gradient finite flag differs for {run_id}/{epoch}")
        if flag == "true":
            require(l2 >= 0 and math.isfinite(float(gradient["gradient_abs_max"])), f"Invalid finite gradient for {run_id}/{epoch}")
        nonfinite_stems += flag == "false"
    for run_id, config in configs.items():
        recorded = config["recorded_settings"]
        require(skipped[run_id] == recorded["amp_skipped_steps_total"], f"Skipped-update total differs for {run_id}")
        close(history_index[run_id, "1"]["learning_rate"], recorded["learning_rate_initial"], f"Initial learning rate differs for {run_id}")
        close(history_index[run_id, str(recorded["epochs_run"])]["learning_rate"], recorded["learning_rate_final"], f"Final learning rate differs for {run_id}")
    scalar_index = indexed(scalars, ("run_id", "epoch", "term"), "coefficient-gradient summary")
    zabra_runs = {run_id: row for run_id, row in runs.items() if row["activation"] in TERMS}
    expected_scalars = {(run_id, str(epoch), term) for run_id, row in zabra_runs.items() for epoch in range(1, int(sources[run_id]["epochs"]) + 1) for term in TERMS[row["activation"]]}
    require(set(scalar_index) == expected_scalars, "Coefficient-gradient coverage differs")
    for row in scalars:
        same_identity(row, runs[row["run_id"]])
        expected = SITES[row["model"]]
        finite = int(row["finite_sites"])
        nonfinite = int(row["nonfinite_sites"])
        require(int(row["expected_sites"]) == expected and finite + nonfinite == expected and finite >= 0 and nonfinite >= 0, f"Scalar gradient site counts differ for {row['run_id']}/{row['epoch']}")
        values = [row[name] for name in ("minimum_abs_gradient", "median_abs_gradient", "maximum_abs_gradient")]
        if finite:
            numbers = [float(value) for value in values]
            require(all(math.isfinite(value) and value >= 0 for value in numbers) and numbers == sorted(numbers), f"Invalid scalar gradient summary for {row['run_id']}/{row['epoch']}")
        else:
            require(not any(values), f"Values exist without finite scalar samples for {row['run_id']}/{row['epoch']}")
    indexed(coefficients, ("run_id", "site"), "final coefficient site")
    coefficient_counts = defaultdict(int)
    for row in coefficients:
        run_id = row["run_id"]
        require(run_id in zabra_runs, f"Unexpected coefficient run {run_id}")
        same_identity(row, runs[run_id])
        coefficient_counts[run_id] += 1
        for raw, effective in (("a", "odd_coefficient"), ("c", "even_coefficient")):
            value = float(row[raw])
            bounded = float(row[effective])
            require(math.isfinite(value) and math.isfinite(bounded) and abs(bounded) <= 1, f"Invalid coefficient for {run_id}/{row['site']}")
            require(abs(math.tanh(value) - bounded) < 1e-7, f"Coefficient transform differs for {run_id}/{row['site']}")
            if raw not in TERMS[row["activation"]]:
                require(value == bounded == 0, f"Inactive coefficient differs for {run_id}/{row['site']}")
    require(set(coefficient_counts) == set(zabra_runs), "Final-coefficient run coverage differs")
    require(all(count == SITES[runs[run_id]["model"]] for run_id, count in coefficient_counts.items()), "Final-coefficient site counts differ")
    audits = {key[0]: row for key, row in indexed(tables["additional_source_audit.csv"], ("run_id",), "additional source audit").items()}
    require(set(audits) == set(runs) - set(scores), "Additional-source audit coverage differs")
    for run_id, audit in audits.items():
        same_identity(audit, runs[run_id])
        for name in ("source_result_sha256", "source_history_sha256", "source_gradient_norms_sha256", "source_receipt_sha256"):
            require(audit[name] == sources[run_id][name], f"Additional provenance differs for {run_id}/{name}")
        for name in ("source_config_sha256", "source_experiment_identity_sha256", "source_code_sha256"):
            require(audit[name] == configs[run_id][name], f"Additional config provenance differs for {run_id}/{name}")
        require(audit["source_predictions_sha256"] == runs[run_id]["source_predictions_sha256"], f"Additional prediction provenance differs for {run_id}")
        for name, value in audit.items():
            if name.endswith("sha256"):
                valid_hash(value, f"{run_id}/{name}")
        require(audit["archive_subset_matches_source"].lower() == "true" and audit["receipt_verified"].lower() == "true", f"Additional source audit is incomplete for {run_id}")
    return {
        "scored_runs": len(runs),
        "original_scored_runs": len(scores),
        "additional_scored_runs": len(audits),
        "training_epochs": len(history),
        "stem_gradient_samples": len(stems),
        "nonfinite_stem_gradients": nonfinite_stems,
        "coefficient_gradient_epoch_terms": len(scalars),
        "coefficient_gradient_samples": sum(int(row["expected_sites"]) for row in scalars),
        "nonfinite_coefficient_gradients": sum(int(row["nonfinite_sites"]) for row in scalars),
        "final_coefficient_sites": len(coefficients),
        "runs_with_receipt_checks": sum(bool(row["receipt_checked_files"]) for row in source_rows),
        "validation_f1_records": sum(bool(row["val_f1"]) for row in history),
        "source_scores_sha256": digest(scores_path),
        "prediction_manifest_sha256": digest(manifest_path),
        "run_configs_sha256": digest(configs_path),
        "training_files": file_manifest["file_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=Path("results/training"))
    parser.add_argument("--scores", type=Path, default=Path("results/run_scores.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("results/prediction_manifest.csv"))
    parser.add_argument("--configs", type=Path, default=Path("results/configurations/run_configs.jsonl"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.diagnostics, args.scores, args.manifest, args.configs)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")

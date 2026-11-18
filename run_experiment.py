"""Run the ZABRA comparison or component panel under the published protocol."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocols" / "training.json").read_text(encoding="utf-8"))
TRAINER = ROOT / "benchmark" / "train_benchmark.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["main", "ablation"], default="main")
    parser.add_argument("--dataset", nargs="+", choices=PROTOCOL["datasets"])
    parser.add_argument("--model", nargs="+", choices=PROTOCOL["models"])
    parser.add_argument("--activation", nargs="+", choices=["ZABRA", *PROTOCOL["comparison_activations"],
                                                           *PROTOCOL["ablation"]["activations"]])
    parser.add_argument("--seed", type=int, nargs="+", choices=PROTOCOL["seeds"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--pause-file", type=Path, help="Save a checkpoint and stop at the next epoch boundary.")
    parser.add_argument("--dry-run", action="store_true", help="List selected runs without training or downloading.")
    args = parser.parse_args()
    if args.panel == "ablation":
        ablation = PROTOCOL["ablation"]
        args.dataset = args.dataset or [ablation["dataset"]]
        args.model = args.model or [ablation["model"]]
        args.activation = args.activation or ablation["activations"]
        if (args.dataset != [ablation["dataset"]] or args.model != [ablation["model"]] or
                not set(args.activation) <= set(ablation["activations"])):
            parser.error("The ablation panel uses CIFAR-100, ResNet18 and the odd-only/even-only variants.")
    else:
        args.dataset = args.dataset or list(PROTOCOL["datasets"])
        args.model = args.model or PROTOCOL["models"]
        args.activation = args.activation or ["ZABRA"]
        if set(args.activation) & set(PROTOCOL["ablation"]["activations"]):
            parser.error("Use --panel ablation for the two component arms.")
    args.seed = args.seed or PROTOCOL["seeds"]
    for items in (args.dataset, args.model, args.activation, args.seed):
        if len(items) != len(set(items)):
            parser.error("Each selection value must appear only once.")
    return args


def training_command(args, dataset, model, activation, seed):
    profile = "imagenet100" if dataset == "imagenet100" else "small_image"
    flags = PROTOCOL["training_arguments"][profile]
    output_dir = args.output_dir / dataset / model / activation / str(seed)
    activation = PROTOCOL["engine_activation_names"].get(activation, activation)
    output_path = str(output_dir.resolve())
    if os.name == "nt" and not output_path.startswith("\\\\?\\"):
        output_path = "\\\\?\\UNC\\" + output_path[2:] if output_path.startswith("\\\\") else "\\\\?\\" + output_path
    command = [sys.executable, "-u", str(TRAINER), "--datasets", dataset, "--models", model,
               "--activations", activation, "--seeds", str(seed), "--seed_order", *map(str, PROTOCOL["seeds"]),
               *flags, "--data_dir", str(args.data_dir.resolve()), "--output_dir", output_path]
    if args.pause_file:
        command += ["--pause_file", str(args.pause_file.resolve())]
    return command


def main():
    args = parse_args()
    jobs = list(itertools.product(args.dataset, args.model, args.activation, args.seed))
    print(f"{len(jobs)} runs, 100 epochs per run", flush=True)
    for dataset, model, activation, seed in jobs:
        print(f"{dataset}  {model}  {activation}  seed={seed}", flush=True)
        if not args.dry_run:
            rc = subprocess.run(training_command(args, dataset, model, activation, seed)).returncode
            if rc:
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())

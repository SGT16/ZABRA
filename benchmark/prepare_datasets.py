"""Fetch benchmark data or verify an existing copy against the published splits."""

import argparse
import json
import os
from pathlib import Path
import sys

from activation_benchmark.data import dataset_manifest, load_bundle


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "training.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", choices=PROTOCOL["datasets"], default=list(PROTOCOL["datasets"]))
    parser.add_argument("--verify-only", action="store_true", help="Read a previously downloaded local cache.")
    args = parser.parse_args()
    if args.verify_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    for name in dict.fromkeys(args.datasets):
        bundle = load_bundle(name, args.data_dir, med_size=64, download=not args.verify_only)
        manifest = dataset_manifest(bundle)
        target = PROTOCOL["datasets"][name]
        values = {"train_count": manifest["train"] + manifest["validation"], "test_count": manifest["test"],
                  "test_labels_sha256": manifest["test_labels_sha256"],
                  "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"]}
        for key, value in values.items():
            if value != target[key]:
                raise ValueError(f"{name} {key} differs from the published protocol: {value} (expected {target[key]})")
        print(f"{name}: {values['train_count']} train, {values['test_count']} test, "
              f"fingerprint {values['dataset_fingerprint_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.torch_version import TorchVersion
from torchvision.datasets import CIFAR100


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual, predicted):
    n = np.bincount(actual, minlength=100)
    k = np.bincount(predicted, minlength=100)
    correct = np.bincount(actual[actual == predicted], minlength=100)
    return np.mean(actual == predicted), np.mean(2 * correct / (n + k))


def bytes_sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_digest(state):
    sha = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        shape = ",".join(map(str, array.shape))
        for data in (name.encode(), str(tensor.dtype).encode(), shape.encode(), array.tobytes()):
            sha.update(len(data).to_bytes(8, "little"))
            sha.update(data)
    return sha.hexdigest()


def source_inputs(root, row, labels):
    seed = int(row["seed"])
    folder = root / f"cifar100-zabra-resnet18-{seed}"
    receipt = json.loads((folder / "receipt.json").read_bytes())
    hashes = {r["file"]: r["sha256"] for r in receipt["files"]}
    assert digest(folder / "done.json") == row["source_result_sha256"]
    weights = folder / "final_weights.pt"
    assert digest(weights) == hashes["final_weights.pt"]
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(weights, weights_only=True, map_location="cpu", mmap=True)
    assert checkpoint["metadata"]["seed"] == seed and checkpoint["metadata"]["epochs_run"] == 100
    archived = folder / "test_predictions.npz"
    assert digest(archived) == row["source_predictions_sha256"]
    with np.load(archived, allow_pickle=False) as pred:
        np.testing.assert_array_equal(pred["y_true"], labels)
        reference = pred["probabilities"].argmax(1)
    return checkpoint["state_dict"], reference, hashes["final_weights.pt"]


def bundled_inputs(bundle, row, labels):
    seed = int(row["seed"])
    with ZipFile(bundle) as archive:
        entries = {name: value for value, name in
                   (line.split("  ", 1) for line in archive.read("SHA256SUMS.txt").decode().splitlines())}
        manifest = archive.read("manifest.json")
        assert bytes_sha(manifest) == entries["manifest.json"]
        provenance = json.loads(manifest)
        assert provenance["test_labels_sha256"] == bytes_sha(labels.astype("<i8").tobytes())
        record = next(r for r in provenance["runs"] if r["seed"] == seed)
        assert record["source_result_sha256"] == row["source_result_sha256"]
        assert record["source_predictions_sha256"] == row["source_predictions_sha256"]
        weights = archive.read(record["weights_file"])
        assert bytes_sha(weights) == entries[record["weights_file"]] == record["weights_sha256"]
        checkpoint = torch.load(io.BytesIO(weights), weights_only=True, map_location="cpu")
        assert set(checkpoint) == {"seed", "epoch", "state_dict"}
        assert checkpoint["seed"] == record["seed"] == seed and checkpoint["epoch"] == record["epoch"] == 100
        state = checkpoint["state_dict"]
        assert tensor_digest(state) == record["source_state_dict_sha256"]
        raw = archive.read(record["classes_file"])
        assert bytes_sha(raw) == entries[record["classes_file"]] == record["source_argmax_uint8_sha256"]
        reference = np.frombuffer(raw, dtype=np.uint8)
        assert reference.shape == labels.shape and np.all(reference < 100)
        acc, score = metrics(labels, reference)
        assert abs(acc - float(row["top1"])) < 1e-12 and abs(score - float(row["macro_f1"])) < 1e-12
    return state, reference, record["source_weights_sha256"]


def evaluate(args, labels, loader, released, build_model):
    site_results = []
    classification = []
    sites = ["relu", "layer2.1.act2", "layer4.1.act2"]
    for row in sorted(released, key=lambda r: int(r["seed"])):
        seed = int(row["seed"])
        if args.weights_bundle:
            state, reference, source_hash = bundled_inputs(args.weights_bundle, row, labels)
        else:
            state, reference, source_hash = source_inputs(args.weights_root, row, labels)
        model = build_model("ResNet18", 3, 100, "ZABRA", pretrained=False, small_input_stem=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        accum = {site: [0, 0.0, 0.0, 0] for site in sites}
        states = {}
        def on_site(site):
            def collect(module, inputs, output):
                x = inputs[0]
                plain = torch.relu(x)
                correction = (output - plain).abs()
                n, total, peak, neg = accum[site]
                accum[site] = [n + x.numel(), total + correction.sum(dtype=torch.float64).item(),
                               max(peak, correction.max().item()), neg + (x < 0).sum().item()]
                if site == sites[-1]:
                    states["native"] = output
                    states["reset"] = plain
            return collect
        hooks = [model.get_submodule(site).register_forward_hook(on_site(site)) for site in sites]
        preds, reset_preds, seen = [], [], []
        with torch.inference_mode():
            for x, y in loader:
                logits = model(x)
                propagated = model.fc(model.avgpool(states["native"]).flatten(1))
                torch.testing.assert_close(logits, propagated, rtol=1e-6, atol=1e-6)
                reset = model.fc(model.avgpool(states["reset"]).flatten(1))
                preds.extend(logits.argmax(1).tolist())
                reset_preds.extend(reset.argmax(1).tolist())
                seen.extend(y.tolist())
        for hook in hooks:
            hook.remove()
        np.testing.assert_array_equal(seen, labels)
        f = np.array(preds)
        g = np.array(reset_preds)
        top, full_f1 = metrics(labels, f)
        atop, alt_f1 = metrics(labels, g)
        source = {"seed": seed, "source_weights_sha256": source_hash}
        classification.append({**source, "n_images": len(labels), "fp32_full_top1": top,
              "fp32_last_relu_top1": atop, "fp32_full_macro_f1": full_f1, "fp32_last_relu_macro_f1": alt_f1,
              "last_relu_minus_full_f1_pp": 100 * (alt_f1 - full_f1), "changed_predictions": int(sum(g != f)),
              "archived_full_vs_fp32_changed_predictions": int(sum(reference != f)),
              "archived_macro_f1": row["macro_f1"], "torch_version": str(torch.__version__)})
        for site in sites:
            module = model.get_submodule(site)
            envelope = abs(float(module.a.detach().tanh())) / 2 + abs(float(module.c.detach().tanh())) / 4
            n, total, peak, neg = accum[site]
            assert total / n <= peak <= envelope + 1e-5
            site_results.append({**source, "site": site, "n_images": len(labels), "scalar_inputs": n,
               "negative_input_fraction": neg / n, "mean_absolute_correction": total / n,
               "empirical_maximum": peak, "analytical_maximum": envelope,
               "mean_as_fraction_of_maximum": total / n / envelope})
        print(f"{seed}: full {100*full_f1:.4f}, final-site ReLU {100*alt_f1:.4f}", flush=True)
    return site_results, classification


def main():
    p = argparse.ArgumentParser()
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--weights-root", type=Path, help="Directory of the original run records")
    inputs.add_argument("--weights-bundle", type=Path, help="Portable ZABRA CIFAR-100 ResNet18 checkpoint ZIP")
    for key in ["results-dir", "engine-dir", "data-root", "out"]:
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    sys.path.insert(0, str(args.engine_dir.resolve()))
    from activation_benchmark.models import build_model
    from activation_benchmark.data import build_transforms
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    test_tf = build_transforms("cifar100", 32, 3, "standard")[1]
    ds = CIFAR100(args.data_root, train=False, transform=test_tf, download=False)
    labels = np.asarray(ds.targets, dtype="<i8")
    proto = json.loads((args.results_dir / "protocol.json").read_text(encoding="utf-8"))
    with (args.results_dir / "run_scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(labels) == 10000 and hashlib.sha256(labels.tobytes()).hexdigest() == proto["datasets"]["cifar100"]["test_labels_sha256"]
    chosen = [r for r in rows if (r["dataset"], r["model"], r["activation"]) == ("cifar100","ResNet18","ZABRA")]
    assert {int(r["seed"]) for r in chosen} == set(proto["seeds"])
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    a, b = evaluate(args, labels, loader, chosen, build_model)
    args.out.mkdir(exist_ok=True, parents=True)
    write_rows(args.out / "observed_site_responses.csv", a)
    write_rows(args.out / "frozen_site_intervention.csv", b)


if __name__ == "__main__":
    main()

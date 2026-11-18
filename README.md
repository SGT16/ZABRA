# ZABRA

This study compares ZABRA with other activation functions under a matched experimental protocol. This repository provides the training code, run configurations and recorded results for that comparison.

Let $u(x)=x/(1+x^2)$. ZABRA is

$$
f(x)=\max(0,x)+\tanh(a)\,u(x)+\tanh(c)\,u(x)^2.
$$

Each activation site learns scalars $a$ and $c$, initialized to zero. The odd-only variant fixes $c=0$; the even-only variant fixes $a=0$. [Implementation](benchmark/activation_benchmark/activations.py)

## Setup and training

The experiment uses CIFAR-100, ImageNet-100, DermaMNIST and OrganSMNIST with ResNet18, MobileNetV2 and EfficientNetV2-S. Models train from random initialization for 100 epochs and are evaluated on the test set at the final checkpoint. The matched seeds are 936137, 529212, 448132, 942529 and 110107. [Training protocol](protocols/training.json)

Use Windows, Python 3.14 and a CUDA-compatible GPU. In an activated environment, install the dependencies and prepare the datasets:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python benchmark/prepare_datasets.py --data-dir data
```

Data preparation checks dataset fingerprints and test-label order. The data sources are torchvision CIFAR-100, the pinned `clane9/imagenet-100` split and the official 64-pixel MedMNIST archives. Dataset use follows the providers' terms.

```powershell
python run_experiment.py --panel main
python run_experiment.py --panel ablation
python run_experiment.py --dataset cifar100 --model ResNet18 --activation ZABRA ReLU --seed 942529
```

Add `--dry-run` to list jobs without training. Use `--help` for the supported comparison activations. Outputs are written under `outputs/<dataset>/<model>/<activation>/<seed>/`.

## Recorded results

The released records contain **685 scored runs** and **68,500 training epochs**. These comprise 675 comparison runs and 10 component-ablation runs. Five additional PowLU runs ended in documented numerical failures. Available results retain their original run identifiers and source-file hashes.

- [All 685 test scores and prediction-file identities](results/prediction_manifest.csv)
- [137 groups of five matched seeds](results/prediction_metrics_by_setting.csv)
- [Training histories and gradients](results/training/README.md)
- [Per-run configurations](results/configurations/run_configs.jsonl)
- [Ablation results](results/ablation_scores.csv) and [GPU measurements](results/benchmark_cells.csv)
- [Numerical failures](results/unscored_controls.csv)

Twenty comparison settings have no scored result: 11 ACON-C settings and the nine ImageNet-100 settings for GELU and the two Squareplus variants. Their absence is not a measured score of zero.

The [download manifest](results/release_assets.csv) provides direct links, file sizes and SHA-256 checksums for the four prediction ZIP files and the five-checkpoint CIFAR-100/ResNet18 ZABRA archive.

## Verification

Run from the repository root:

```powershell
python analysis/verify_results.py
python analysis/verify_training_diagnostics.py --diagnostics results/training --scores results/run_scores.csv --manifest results/prediction_manifest.csv --configs results/configurations/run_configs.jsonl
python analysis/verify_prediction_bundles.py --bundle-dir downloads --output outputs/recomputed_metrics.csv
```

The first command checks the original 635-run score collection, class counts, paired results and GPU measurements. The second checks the full training records and run configurations. The third recomputes all 685 test scores from the four downloaded prediction archives; it requires NumPy. [File definitions and measurement commands](results/README.md)

The shared training implementation retains its [MIT license notice](benchmark/THIRD_PARTY_LICENSE.txt).

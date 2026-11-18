# Experiment results

The records cover 685 scored trainings: 675 comparison runs and 10 ZABRA component-ablation runs. Every scored setting has five matched seeds. Classification scores are measured at epoch 100 on the recorded test split. Scores use the 0–1 scale unless a column explicitly specifies percent or percentage points.

| File | Contents |
|---|---|
| [prediction_manifest.csv](prediction_manifest.csv) | 685 run identities, test accuracy, macro-F1, macro ROC AUC, original source hashes and prediction archive members |
| [prediction_metrics_by_setting.csv](prediction_metrics_by_setting.csv) | Mean and sample standard deviation for 137 five-seed groups |
| [run_scores.csv](run_scores.csv) | Original 635-run collection, including training times, memory and evaluation precision |
| [class_counts.jsonl](class_counts.jsonl) | Class-ordered supports, prediction counts and true positives for those 635 runs |
| [controls_seed_pairs.csv](controls_seed_pairs.csv) | Same-seed ZABRA-minus-control score differences |
| [ablation_scores.csv](ablation_scores.csv) | Full ZABRA, ReLU, odd-only and even-only results on CIFAR-100/ResNet18 |
| [benchmark_blocks.csv](benchmark_blocks.csv) and [benchmark_cells.csv](benchmark_cells.csv) | Repeated GPU timing measurements and 16 measurement-cell summaries |
| [unscored_controls.csv](unscored_controls.csv) | Five PowLU numerical failures on CIFAR-100/EfficientNetV2-S |
| [training/](training/README.md) | 68,500 recorded epochs, gradient samples, learned coefficients and source hashes |
| [configurations/run_configs.jsonl](configurations/run_configs.jsonl) | Portable scientific configurations for the 685 scored runs, with original configuration hashes |
| [measurements/](measurements/) | Activation-input measurements, frozen-model intervention results and directional error counts |

The 50 additional scored controls are 45 GELU runs across CIFAR-100, DermaMNIST and OrganSMNIST and five ACON-C runs on CIFAR-100/EfficientNetV2-S. They are included in the 685-run manifest and training records. Original run identities distinguish these records from the first 635 runs.

Twenty settings have no scored record: ACON-C on the other 11 dataset–backbone settings, and GELU, fixed Squareplus and learned Squareplus on the three ImageNet-100 backbones. The PowLU failure setting is recorded separately. No score has been imputed for these settings.

## Configurations and checkpoints

Portable configurations retain the run's effective dataset, backbone, activation, seed and training/evaluation settings. Machine-specific paths and unused shared-command defaults are omitted. Original configuration, result and history hashes retain the connection to the archived records.

The Squareplus controls use $(x+\sqrt{x^2+\alpha^2})/2$. The fixed scale is 0.4; the learned scale starts at 0.4 and is constrained to [0.05, 1.5], with one scalar per activation site. The conventional Squareplus parameter is $b=\alpha^2$.

The component experiment has 20 scores: ten full-ZABRA/ReLU scores reused from the main runs and ten separately trained odd-only/even-only scores. Reusing a score in a comparison does not create another training run.

[release_assets.csv](release_assets.csv) records download URLs, byte sizes and SHA-256 values for the four prediction ZIPs and the five-checkpoint ZABRA archive. Predictions retain their original numeric test labels, probabilities and logits. [Dataset and label-order identities](prediction_protocol.json)

## Verification

From the repository root:

```powershell
python analysis/verify_results.py
python analysis/verify_training_diagnostics.py --diagnostics results/training --scores results/run_scores.csv --manifest results/prediction_manifest.csv --configs results/configurations/run_configs.jsonl
python analysis/verify_prediction_bundles.py --bundle-dir downloads --output outputs/recomputed_metrics.csv
```

The prediction verifier checks archive hashes, test-label order and per-run scores. It recomputes macro ROC AUC by averaging one-vs-rest AUC over classes, with average ranks for tied probabilities. The training verifier checks run coverage, epoch sequences, recorded gradients, coefficients and source identities.

## GPU measurements

The recorded eager and CUDA Graph measurements use an NVIDIA RTX A5500, FP16 autocast, synthetic inputs, 20 warm-up calls and ten timing blocks of 20 calls. Whole-model batches are 1 and 128; stem workloads use a 128 × 64 × 32 × 32 input. Memory fields labeled `*_mb` use 1,048,576 bytes per unit.

After training a matched CIFAR-100/ResNet18 pair at seed 942529, run:

```powershell
$zabra = (Get-ChildItem outputs/cifar100/ResNet18/ZABRA/942529 -Recurse -Filter final_weights.pt -File -ErrorAction Stop).FullName
$relu = (Get-ChildItem outputs/cifar100/ResNet18/ReLU/942529 -Recurse -Filter final_weights.pt -File -ErrorAction Stop).FullName
python measure_cost.py --engine-dir benchmark --zabra-weights "$zabra" --relu-weights "$relu" --output-dir outputs/cost
```

To repeat the saved CIFAR-100 activation-input and frozen-model measurements with the downloaded ZABRA weights and prepared CIFAR-100 data:

```powershell
python analysis/checkpoint_response.py --weights-bundle downloads/ZABRA_CIFAR100_ResNet18_Agirliklar.zip --results-dir results --engine-dir benchmark --data-root data/torchvision --out outputs/measurements
```

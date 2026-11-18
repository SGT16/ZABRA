# Training records

Recorded histories, sampled gradients and final activation coefficients for the 685 scored runs. Each run has 100 epochs. Join tables by `run_id` to [the score manifest](../prediction_manifest.csv) and [the run configurations](../configurations/run_configs.jsonl).

| File | Records | Contents |
|---|---:|---|
| `training_history.csv` | 68,500 | Per-epoch training loss, learning rate, duration, AMP scale and skipped updates |
| `stem_gradients.csv` | 68,500 | First-convolution gradient and parameter norms at the first training minibatch of each epoch |
| `coefficient_gradient_epochs.csv` | 13,000 | Per-epoch summaries of trainable ZABRA coefficient gradients |
| `final_coefficients_verified.csv` | 3,250 | Final raw coefficients and their hyperbolic tangents at all ZABRA sites |
| `training_source_hashes.csv` | 685 | Original result, history, gradient and coefficient hashes, with receipt coverage where available |
| `additional_source_audit.csv` | 50 | Source hash bindings and receipt checks for the additional controls |
| `training_files.json` | — | SHA-256 values for the exported training tables |
| `validation.json` | — | Verification results for the released records |

The first 635 runs contribute 63,500 epochs. The additional 45 GELU and five ACON-C controls contribute 5,000 epochs. All records are available in this repository. Original source-file identities are retained; the `stable.` prefix identifies the additional control records.

The history fields `val_f1`, `acc` and `loss` retain the original validation fields. These runs use final-epoch evaluation on the test split; the final test scores are in the score manifest. Training loss is recorded separately as `train_loss`.

Gradients are recorded after AMP unscaling at batch index zero. The first convolution is `conv1.weight` for ResNet18 and `features.0.0.weight` for MobileNetV2 and EfficientNetV2-S. Source flags record nonfinite samples. Coefficient-gradient rows give site counts and the minimum, median and maximum absolute gradient across finite sites.

Final coefficient rows contain raw `a` and `c` and effective `tanh(a)` and `tanh(c)`. The odd-only and even-only arms retain only their respective trainable term. Activation-input and frozen-model measurements are in [measurements/](../measurements/).

## Verification

From the repository root, using the Python standard library:

```powershell
python analysis/verify_training_diagnostics.py --diagnostics results/training --scores results/run_scores.csv --manifest results/prediction_manifest.csv --configs results/configurations/run_configs.jsonl
```

The verifier checks table hashes, exact run coverage, consecutive epochs, source-result identities, configuration consistency, gradient sampling and coefficient transforms. The `receipt_checked_files` column names only the original files covered by each stored receipt.

To export the original 635-run diagnostics from locally available run directories, provide a CSV with `run_id,result_path`; each result file must have its recorded companion history and gradient files:

```powershell
python analysis/export_training_diagnostics.py --scores results/run_scores.csv --source-index SOURCE_INDEX.csv --output outputs/training
```

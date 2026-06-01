# Chronos-2

[Amazon Chronos-2](https://huggingface.co/amazon/chronos-2) zero-shot time-series forecaster, used in Energy-EVA 2.0 leaderboard with native covariate support.

## Install

```bash
pip install chronos-forecasting
```

A version exposing `chronos.Chronos2Pipeline` is required.

## Model weights

Download the checkpoint from Hugging Face and place it under the directory pointed to by `time_series_portal.config.MODEL_STORAGE_PATH`:

```
${MODEL_STORAGE_PATH}/amazon__chronos-2/
```

## Registry

| Registered name | Adapter | Predict path |
|---|---|---|
| `chronos-2` | `Chronos2CovarAdapter` | `native_covar_predict_with_model` |

The leaderboard value reported in the project README is produced by this configuration (native covariate inputs). If a dataset has no `known_dynamic_columns`, the adapter degrades to a target-only forecast for that task automatically.

## Invocation

```bash
python time_series_portal/evaluation.py \
    --dataset_path PATH/TO/DATASETS \
    --target_path PATH/TO/RESULTS \
    --scene wind solar load \
    --model chronos-2
```

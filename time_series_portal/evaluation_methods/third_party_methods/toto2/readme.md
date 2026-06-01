# Toto 2.0 (2.5B)

[DataDog Toto-2.0-2.5B](https://huggingface.co/datadog/toto-2.0-2.5B) zero-shot time-series forecaster with native covariate support, used in Energy-EVA 2.0 leaderboard.

## Prerequisites

Install `toto2` from the [GitHub repository](https://github.com/DataDog/toto) (the package is not distributed on PyPI), and download the weights from Hugging Face. Place the checkpoint directory under `time_series_portal.config.MODEL_STORAGE_PATH`:

```
${MODEL_STORAGE_PATH}/datadog__toto-2.0-2.5B/
```

## Registry

| Registered name | Adapter | Predict path |
|---|---|---|
| `toto_2.0_2.5B` | `Toto2CovarAdapter` | `native_covar_predict_with_model` |

The leaderboard value reported in the project README is produced by this configuration (native covariate inputs). When a dataset has no `known_dynamic_columns`, the adapter forecasts on the target alone.

## Invocation

```bash
python time_series_portal/evaluation.py \
    --dataset_path PATH/TO/DATASETS \
    --target_path PATH/TO/RESULTS \
    --scene wind solar load \
    --model toto_2.0_2.5B
```

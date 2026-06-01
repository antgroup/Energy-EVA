# Moirai 2.0 (R, small)

[Salesforce Moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) zero-shot time-series forecaster, used in Energy-EVA 2.0 leaderboard.

## Install

```bash
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
```

The `uni2ts.model.moirai2` subpackage must be available (Moirai 2.0 series uses different module paths from 1.x).

## Model weights

Place the checkpoint under the directory pointed to by `time_series_portal.config.MODEL_STORAGE_PATH`:

```
${MODEL_STORAGE_PATH}/salesforce__moirai-2.0-R-small/
```

## Registry

| Registered name | Adapter | Predict path |
|---|---|---|
| `moirai_2.0_R_small` | `Moirai2Adapter` | `univar_predict_with_model` |

Moirai 2.0 outputs 9 quantiles; the adapter selects index 4 (median) as the point forecast.

On Apple-silicon (MPS) the adapter falls back to CPU automatically due to upstream `cummax` / `float64` compatibility limitations.

## Invocation

```bash
python time_series_portal/evaluation.py \
    --dataset_path PATH/TO/DATASETS \
    --target_path PATH/TO/RESULTS \
    --scene wind solar load \
    --model moirai_2.0_R_small
```

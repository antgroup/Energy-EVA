# Simons

[Simons](https://github.com/Abel-ai-lab/Simons) (Abel AI Lab) is a family of four open zero-shot time-series forecasters with native known-future covariate support: [Simons-25M](https://huggingface.co/abel-lab/Simons-25M), [Simons-60M](https://huggingface.co/abel-lab/Simons-60M), [Simons-178M](https://huggingface.co/abel-lab/Simons-178M) and [Simons-297M](https://huggingface.co/abel-lab/Simons-297M). Code: Apache-2.0; weights: OpenMDW-1.1.

## Install

```bash
pip install "git+https://github.com/Abel-ai-lab/Simons@v0.1.1"
```

## Model weights

The weights are downloaded from the Hugging Face Hub on first use. To use a local copy, place the checkpoint directory (`config.json`, `model.safetensors`) under `time_series_portal.config.MODEL_STORAGE_PATH`:

```
${MODEL_STORAGE_PATH}/abel-lab__Simons-297M/
```

## Registry

| Registered name | Adapter | Predict path |
|---|---|---|
| `simons-25m`, `simons-60m`, `simons-178m`, `simons-297m` | `SimonsAdapter` | `native_covar_predict_with_model` |
| `simons-25m_univar`, `simons-60m_univar`, `simons-178m_univar`, `simons-297m_univar` | `SimonsAdapter` | `univar_predict_with_model` |

The model is used zero-shot with its released defaults and is not fine-tuned on the benchmark. `simons-<size>` passes the task's `known_dynamic_columns` to the model as known-future covariates (context and horizon); `simons-<size>_univar` forecasts from the target alone and is the name to use for scenes without `known_dynamic_columns`. The point forecast is the median (quantile 0.5).

## Invocation

```bash
python time_series_portal/evaluation.py \
    --dataset_path PATH/TO/DATASETS \
    --target_path PATH/TO/RESULTS \
    --scene wind solar price \
    --model simons-297m
```

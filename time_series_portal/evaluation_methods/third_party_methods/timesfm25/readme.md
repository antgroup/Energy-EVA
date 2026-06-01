# TimesFM 2.5 (200M, PyTorch)

[Google TimesFM-2.5-200M-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) zero-shot time-series forecaster with early-fusion XReg covariate support, used in Energy-EVA 2.0 leaderboard.

## Install

```bash
pip install git+https://github.com/google-research/timesfm.git
```

A version exposing `timesfm.TimesFM_2p5_200M_torch` and `timesfm.ForecastConfig` is required (the 2.0 API is incompatible).

## Model weights

Place the checkpoint directory under `time_series_portal.config.MODEL_STORAGE_PATH`:

```
${MODEL_STORAGE_PATH}/google__timesfm-2.5-200m-pytorch/
```

## Registry

| Registered name | Adapter | Predict path |
|---|---|---|
| `timesfm2.5_xreg_early` | `TimesFM25Adapter` (covariates=True, fusion=early) | `covariates_predict_with_model` |

The adapter compiles the model lazily per `(horizon, max_context, return_backcast)` triple and caches the compiled state, so changing horizon across tasks is supported without manual reset. Context is left-cropped to the largest multiple of 32 (input patch length) to avoid upstream NaN issues caused by fully-masked leading patches.

## Invocation

```bash
python time_series_portal/evaluation.py \
    --dataset_path PATH/TO/DATASETS \
    --target_path PATH/TO/RESULTS \
    --scene wind solar load \
    --model timesfm2.5_xreg_early
```

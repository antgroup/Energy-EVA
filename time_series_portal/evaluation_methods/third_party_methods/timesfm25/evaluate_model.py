import os
from typing import Dict, List, Literal, Optional

import fev
import numpy as np
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils import registry, PersistenceUtil
from time_series_portal.evaluation_utils.unify_predict_methods import covariates_predict_with_model


class TimesFM25Adapter(CallableAdapter):
    """TimesFM 2.5 adapter.

    Mirrors the official README and `api_reference.md`:
    - point_forecast is already the median; no quantile slicing needed.
    - xreg path requires `return_backcast=True` (enforced upstream).
    - `dynamic_numerical_covariates` must be `dict[str, list[np.ndarray]]`,
      with each covariate length equal to context + horizon.

    NaN guard: `max_context` must be <= the actual context length and a multiple
    of `INPUT_PATCH_LEN` (32). TimesFM 2.5's `forecast()` left-pads shorter
    inputs with zeros and masks them; `decode()` returns NaN when a leading
    patch is fully masked. We align the model's `max_context` down to a
    multiple of 32 of the actual context length and also crop the input to the
    same length to bypass the padding.
    """

    INPUT_PATCH_LEN = 32  # see TimesFM_2p5_200M_Definition in upstream

    def __init__(self,
                 _model_name: str,
                 _enable_covariates: bool = False,
                 _fusion_mode: Literal['early', 'late'] = 'early',
                 **kwargs):
        super().__init__()
        try:
            from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
        except ImportError:
            raise ImportError('try to install timesfm from "https://github.com/google-research/timesfm"')
        if _model_name != "timesfm2.5":
            raise ValueError(f"{_model_name} is not supported.")
        model_path = os.path.join(MODEL_STORAGE_PATH, 'google__timesfm-2.5-200m-pytorch')
        self.model = TimesFM_2p5_200M_torch.from_pretrained(model_path)
        self._ForecastConfig = ForecastConfig
        self._compiled_key: Optional[tuple] = None  # (horizon, max_context, return_backcast)
        self.enable_covariates = _enable_covariates
        self.fusion_mode = _fusion_mode

    def _aligned_context_len(self, ctx_len: int) -> int:
        aligned = (ctx_len // self.INPUT_PATCH_LEN) * self.INPUT_PATCH_LEN
        return max(aligned, self.INPUT_PATCH_LEN)

    def _ensure_compiled(self, horizon: int, max_context: int, return_backcast: bool) -> None:
        key = (horizon, max_context, return_backcast)
        if self._compiled_key == key:
            return
        self.model.compile(self._ForecastConfig(
            max_context=max_context,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=32,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=return_backcast,
        ))
        self._compiled_key = key

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        device = _input['context_target'].device
        context_np = (_input['context_target']
                      .cpu().squeeze(1).squeeze(0).numpy().astype(np.float32))
        aligned = self._aligned_context_len(len(context_np))
        context_np = context_np[-aligned:]
        use_covariates = self.enable_covariates and 'context_covariates' in _input
        self._ensure_compiled(_prediction_length, aligned, return_backcast=use_covariates)

        if use_covariates:
            n_covar = _input['context_covariates'].shape[1]
            dynamic_numerical_covariates = {
                f'covar_{i}': [
                    np.concatenate([
                        _input['context_covariates'][0, i, -aligned:].cpu().numpy().astype(np.float32),
                        _input['future_covariates'][0, i, :].cpu().numpy().astype(np.float32),
                    ])
                ]
                for i in range(n_covar)
            }
            fusion_mode = 'xreg + timesfm' if self.fusion_mode == 'early' else 'timesfm + xreg'
            point_outputs, _quantile_outputs = self.model.forecast_with_covariates(
                inputs=[context_np],
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                dynamic_categorical_covariates={},
                static_numerical_covariates={},
                static_categorical_covariates={},
                xreg_mode=fusion_mode,
                normalize_xreg_target_per_input=True,
                ridge=0.0,
                force_on_cpu=False,
            )
            result = np.asarray(point_outputs[0], dtype=np.float32)
        else:
            point_forecast, _quantile_forecast = self.model.forecast(
                horizon=_prediction_length,
                inputs=[context_np],
            )
            result = np.asarray(point_forecast[0], dtype=np.float32)

        return {'prediction': torch.from_numpy(result).to(device=device)}


@registry.register("timesfm2.5_xreg_early")
def timesfm2_5_xreg_early(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TimesFM25Adapter(
        'timesfm2.5',
        _enable_covariates=True,
        _fusion_mode='early',
        **kwargs,
    )
    return covariates_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

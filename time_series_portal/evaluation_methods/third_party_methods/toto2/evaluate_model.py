import os
from typing import List, Dict, Union

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils import registry, PersistenceUtil
from time_series_portal.evaluation_utils.unify_predict_methods import native_covar_predict_with_model


class Toto2CovarAdapter(CallableAdapter):
    """Toto 2.0 adapter with native covariate support via `known_dynamic`."""

    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 **kwargs,
                 ):
        super().__init__()
        try:
            from toto2 import Toto2Model
        except ImportError:
            raise ImportError('try to install toto2 from "https://github.com/DataDog/toto"')
        if _model_name != 'toto_2.0_2.5B':
            raise NotImplementedError(f'Unknown model: {_model_name}')
        local_dir = os.path.join(MODEL_STORAGE_PATH, 'datadog__toto-2.0-2.5B')
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(
                f'Toto-2 weights not found at {local_dir}. '
                f'Download `datadog/toto-2.0-2.5B` from Hugging Face and place it under MODEL_STORAGE_PATH.'
            )
        self.model = Toto2Model.from_pretrained(local_dir)
        self.model.to(device).eval()
        self.device = device
        self.model_name = _model_name

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        target = _input['context_target']  # (B, 1, T)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        batch_size, n_var = target.shape[0], target.shape[1]
        series_ids = torch.zeros(batch_size, n_var, dtype=torch.long, device=self.device)

        inputs = {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

        context_covariates = _input.get('context_covariates', None)  # (B, D, T)
        future_covariates = _input.get('future_covariates', None)    # (B, D, H)
        known_dynamic_columns = _input.get('known_dynamic_columns', [])

        if (context_covariates is not None
                and future_covariates is not None
                and len(known_dynamic_columns) > 0):
            known_dynamic = torch.cat([context_covariates, future_covariates], dim=-1)
            known_dynamic_mask = torch.ones_like(known_dynamic, dtype=torch.bool)
            known_dynamic_series_ids = torch.zeros(
                batch_size, known_dynamic.shape[1], dtype=torch.long, device=self.device,
            )
            inputs["known_dynamic"] = known_dynamic
            inputs["known_dynamic_mask"] = known_dynamic_mask
            inputs["known_dynamic_series_ids"] = known_dynamic_series_ids

        quantiles = self.model.forecast(inputs, horizon=_prediction_length)
        # quantiles: (9, B, n_var, H), median at index 4
        median = quantiles[4, 0, 0, :]
        return {'prediction': median.to(device=target.device)}


@registry.register("toto_2.0_2.5B")
def toto_2_0_2_5b(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = Toto2CovarAdapter('toto_2.0_2.5B', **kwargs)
    return native_covar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

import os
from functools import partial
from typing import Dict, List, Union

import fev
import numpy as np
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import (
    native_covar_predict_with_model,
    univar_predict_with_model,
)

# registered size -> Hugging Face repository under `abel-lab`
SIMONS_MODELS = {
    '25m': 'Simons-25M',
    '60m': 'Simons-60M',
    '178m': 'Simons-178M',
    '297m': 'Simons-297M',
}


class SimonsAdapter(CallableAdapter):
    """Simons adapter: zero-shot, released defaults, native known-future covariates when given."""

    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 **kwargs,
                 ):
        super().__init__()
        try:
            from simons import Simons
        except ImportError:
            raise ImportError('try to install simons from "https://github.com/Abel-ai-lab/Simons"')
        if _model_name not in SIMONS_MODELS:
            raise NotImplementedError(f'Unknown model: {_model_name}')
        repository = SIMONS_MODELS[_model_name]
        # a local copy under MODEL_STORAGE_PATH is used when present, else the Hugging Face Hub
        model_path = os.path.join(MODEL_STORAGE_PATH, f'abel-lab__{repository}')
        if not os.path.isdir(model_path):
            model_path = f'abel-lab/{repository}'
        self.model = Simons.from_pretrained(model_path, device=device)
        self.device = device
        self.model_name = _model_name

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        from simons import QUANTILES, forecast_series

        context_target = _input['context_target']  # (B, 1, T)
        context_covariates = _input.get('context_covariates', None)  # (B, D, T)
        future_covariates = _input.get('future_covariates', None)    # (B, D, H)

        series = [m_target.cpu().numpy() for m_target in context_target[:, 0]]
        covariates = None
        if context_covariates is not None and future_covariates is not None:
            # known-future covariates span the context and the horizon: (D, T + H) per series
            known_dynamic = torch.cat([context_covariates, future_covariates], dim=-1)
            covariates = [m_covariates.cpu().numpy() for m_covariates in known_dynamic]

        # per series: (H, 9) quantiles 0.1 .. 0.9; the point forecast is the median
        quantiles = forecast_series(self.model, series, _prediction_length, future_covariates=covariates)
        median = np.stack(quantiles)[..., QUANTILES.index(0.5)]  # (B, H)
        return {'prediction': torch.from_numpy(median).unsqueeze(1).to(device=context_target.device)}


def _simons(_size: str, _with_covariates: bool,
            _tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = SimonsAdapter(_size, **kwargs)
    predict_method = native_covar_predict_with_model if _with_covariates else univar_predict_with_model
    return predict_method(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )


for m_size in SIMONS_MODELS:
    registry.register_function(f'simons-{m_size}', partial(_simons, m_size, True))
    registry.register_function(f'simons-{m_size}_univar', partial(_simons, m_size, False))

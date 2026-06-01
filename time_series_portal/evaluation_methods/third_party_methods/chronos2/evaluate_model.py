import os
from typing import Dict, List

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import native_covar_predict_with_model


class Chronos2Adapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device='cuda:0',
                 torch_dtype='bfloat16',
                 **kwargs
                 ):
        super().__init__()
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError('try to install chronos from "https://github.com/amazon-science/chronos-forecasting"')
        if _model_name != "chronos-2":
            raise ValueError(f"{_model_name} is not supported.")
        model_path = os.path.join(MODEL_STORAGE_PATH, 'amazon__chronos-2')
        pipeline = Chronos2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        pipeline.model.to(device)
        pipeline.model.eval()
        self.device = device
        self.pipeline = pipeline

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        predict_result = self.pipeline.predict(
            inputs=_input['context_target'].cpu(),
            prediction_length=_prediction_length,
        )
        return {
            'prediction': predict_result[0].mean(dim=1, keepdim=True).to(device=_input['context_target'].device)
        }


class Chronos2CovarAdapter(CallableAdapter):
    """Chronos-2 adapter that uses the model's native covariate interface."""

    def __init__(self,
                 _model_name,
                 device='cuda:0',
                 torch_dtype='bfloat16',
                 **kwargs
                 ):
        super().__init__()
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError('try to install chronos from "https://github.com/amazon-science/chronos-forecasting"')
        if _model_name != "chronos-2":
            raise ValueError(f"{_model_name} is not supported.")
        model_path = os.path.join(MODEL_STORAGE_PATH, 'amazon__chronos-2')
        pipeline = Chronos2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        pipeline.model.to(device)
        pipeline.model.eval()
        self.device = device
        self.pipeline = pipeline

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target']  # (B, 1, T)
        known_dynamic_columns = _input.get('known_dynamic_columns', [])
        context_covariates = _input.get('context_covariates', None)  # (B, D, T)
        future_covariates = _input.get('future_covariates', None)    # (B, D, T_future)

        batch_size = context_target.shape[0]
        inputs = []
        for b in range(batch_size):
            task_dict = {"target": context_target[b, 0].cpu()}
            if context_covariates is not None and len(known_dynamic_columns) > 0:
                past_cov, future_cov = {}, {}
                for d, name in enumerate(known_dynamic_columns):
                    past_cov[name] = context_covariates[b, d].cpu()
                    if future_covariates is not None and d < future_covariates.shape[1]:
                        future_cov[name] = future_covariates[b, d].cpu()
                task_dict["past_covariates"] = past_cov
                if future_cov:
                    task_dict["future_covariates"] = future_cov
            inputs.append(task_dict)

        predict_result = self.pipeline.predict(
            inputs=inputs,
            prediction_length=_prediction_length,
        )
        return {
            'prediction': predict_result[0].mean(dim=1, keepdim=True).to(device=context_target.device)
        }


@registry.register("chronos-2")
def chronos_2(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = Chronos2CovarAdapter(_model_name='chronos-2', **kwargs)
    return native_covar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

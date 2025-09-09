import os
from typing import Dict, List

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class ChronosAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device='cuda:0',
                 torch_dtype='bfloat16',
                 **kwargs
                 ):
        super().__init__()
        try:
            from chronos import BaseChronosPipeline
        except ImportError:
            raise ImportError('try to install chronos via `pip install chronos-forecasting`')
        if _model_name == "chronos-bolt-base":
            model_path = os.path.join(MODEL_STORAGE_PATH, 'amazon__chronos-bolt-base')
        else:
            raise ValueError(f"{_model_name} is not supported.")
        pipeline = BaseChronosPipeline.from_pretrained(
            model_path, device_map=device, torch_dtype=torch_dtype
        )
        pipeline.model.eval()
        self.pipeline = pipeline

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        predict_result = self.pipeline.predict(
            context=_input['context_target'].squeeze(1),
            prediction_length=_prediction_length,
            limit_prediction_length=False,
        )
        return {
            'prediction': predict_result.mean(dim=1,keepdim=True).to(device=_input['context_target'].device)
        }


@registry.register("chronos-bolt-base")
def chronos_bolt_base(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = ChronosAdapter(_model_name='chronos-bolt-base', **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

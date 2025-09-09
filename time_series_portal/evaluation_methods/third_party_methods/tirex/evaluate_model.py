import os
from typing import List, Dict, Union

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class TiRexAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 num_samples: int = 1,
                 **kwargs,
                 ):
        super().__init__()
        try:
            from tirex.models.tirex import TiRexZero
        except ImportError:
            raise ImportError('try to install tirex from "https://github.com/NX-AI/tirex"')
        if _model_name == 'tirex':
            model = TiRexZero.from_pretrained(
                os.path.join(MODEL_STORAGE_PATH, 'tirex', 'model.ckpt'),
                device=device,
            )
        else:
            raise NotImplementedError
        self.model = model
        self.model_name = _model_name
        self.num_samples = num_samples

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target'].squeeze(1)
        if self.model_name == 'tirex':
            _, result = self.model.forecast(context_target, prediction_length=_prediction_length)
        else:
            raise NotImplementedError
        return {
            'prediction': result.to(device=context_target.device)[0],
        }


@registry.register("tirex")
def tirex(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TiRexAdapter('tirex', **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

import os
from typing import List, Dict, Union

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class SundialAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 num_samples: int = 1,
                 **kwargs,
                 ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError('try to install transformers via:`pip install transformers==4.40.1`')
        if _model_name == 'sundial_base_128m':
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(MODEL_STORAGE_PATH, 'sundial_base_128m'),
                trust_remote_code=True,
                device_map=device
            )
        else:
            raise NotImplementedError
        self.model = model
        self.model_name = _model_name
        self.num_samples = num_samples

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target'].squeeze(1)
        if self.model_name == 'sundial_base_128m':
            output = self.model.generate(context_target, max_new_tokens=_prediction_length,
                                         num_samples=self.num_samples)
        else:
            raise NotImplementedError
        return {
            'prediction': output.median(dim=1)[0].to(device=context_target.device),
        }


@registry.register("sundial_base_128m")
def sundial_base_128m(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = SundialAdapter('sundial_base_128m',num_samples=100, **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

import os
from typing import List, Dict, Union

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class Moirai2Adapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 **kwargs,
                 ):
        super().__init__()
        try:
            from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
        except ImportError:
            raise ImportError('try to install uni2ts from "https://github.com/SalesforceAIResearch/uni2ts"')
        if _model_name != 'moirai-2.0-R-small':
            raise ValueError(f"{_model_name} is not supported.")
        model_path = os.path.join(MODEL_STORAGE_PATH, 'salesforce__moirai-2.0-R-small')
        model = Moirai2Module.from_pretrained(model_path).to(device=device)
        model.eval()
        self.model = model
        self.forecast_class = Moirai2Forecast

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target']
        forecastor = self.forecast_class(
            module=self.model,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_target.shape[-1],
            prediction_length=_prediction_length,
        )
        ct = torch.permute(context_target, (0, 2, 1))
        past_observed_target = torch.ones_like(ct, dtype=torch.bool)
        past_is_pad = torch.zeros_like(ct, dtype=torch.bool).squeeze(-1)
        forecast = forecastor(
            past_target=ct,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        # Moirai 2.0 returns 9 quantiles along dim=1; index 4 is the median.
        return {
            'prediction': forecast[:, 4:5, :].to(device=context_target.device),
        }


@registry.register("moirai_2.0_R_small")
def moirai_2_0_r_small(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = Moirai2Adapter('moirai-2.0-R-small', **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

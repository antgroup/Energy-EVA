import os
from typing import List, Dict, Union

import fev
import torch
from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class MoiraiAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 num_samples: int = 1,
                 **kwargs,
                 ):
        super().__init__()
        try:
            from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
        except ImportError:
            raise ImportError('try to install uni2ts via: `pip install uni2ts`')
        self.forecast_class = MoiraiForecast
        if _model_name == 'moirai-1.1-R-large':
            model_path = os.path.join(MODEL_STORAGE_PATH, 'moirai-1.1-R-large')
            model = MoiraiModule.from_pretrained(model_path).to(device=device)
            model.eval()
        else:
            raise NotImplementedError
        self.model = model
        self.model_name = _model_name
        self.num_samples = num_samples
        self.patch_size = 64

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target']
        if self.model_name.startswith('moirai-'):
            forecastor = self.forecast_class(
                module=self.model,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                context_length=context_target.shape[-1],
                prediction_length=_prediction_length,
                num_samples=self.num_samples,
                patch_size=self.patch_size,
            )
            context_target = torch.permute(context_target, (0, 2, 1))
            past_observed_target = torch.ones_like(context_target, dtype=torch.bool)
            past_is_pad = torch.zeros_like(context_target, dtype=torch.bool).squeeze(-1)
            forecast = forecastor(
                past_target=context_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )
        else:
            raise NotImplementedError
        return {
            'prediction': forecast.median(dim=1)[0].to(device=context_target.device),
        }


@registry.register("moirai_1.1_R_large")
def chronos_bolt_base_xreg_early(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = MoiraiAdapter('moirai-1.1-R-large', num_samples=100, **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

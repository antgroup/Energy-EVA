import os
from typing import List, Dict, Union

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils import registry, PersistenceUtil
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model


class TOTOAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 device: Union[str, torch.device] = "cuda:0",
                 num_samples: int = 1,
                 **kwargs,
                 ):
        super().__init__()
        try:
            from toto.data.util.dataset import MaskedTimeseries
            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto
        except ImportError:
            raise ImportError('try to install toto from "https://github.com/DataDog/toto"')
        self.data_class = MaskedTimeseries
        if _model_name == 'toto_151m':
            toto = Toto.from_pretrained(
                os.path.join(MODEL_STORAGE_PATH, 'toto'),
            )
            toto.to(device)
        else:
            raise NotImplementedError
        self.model = TotoForecaster(toto.model)
        self.device = device
        self.model_name = _model_name
        self.num_samples = num_samples

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        context_target = _input['context_target'].squeeze(0)
        if self.model_name == 'toto_151m':
            inputs = self.data_class(
                series=context_target.to(self.device),
                padding_mask=torch.full_like(context_target, True, dtype=torch.bool).to(self.device),
                id_mask=torch.zeros_like(context_target).to(self.device),
                timestamp_seconds=torch.zeros_like(context_target).to(self.device),
                time_interval_seconds=torch.full((context_target.shape[0],), 60 * 15).to(self.device),
            )
            forecast = self.model.forecast(
                inputs,
                prediction_length=_prediction_length,
                num_samples=self.num_samples,
                samples_per_batch=self.num_samples,
            )
            result = forecast.median.flatten()
        else:
            raise NotImplementedError
        return {
            'prediction': result.to(device=context_target.device),
        }


@registry.register("toto_151m")
def toto_151m(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TOTOAdapter('toto_151m', num_samples=256, **kwargs)
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

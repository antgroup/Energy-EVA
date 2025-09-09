import os
from typing import Dict, List

import fev
import torch

from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from time_series_portal.config import MODEL_STORAGE_PATH
from time_series_portal.evaluation_utils import registry, PersistenceUtil
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model, \
    covariates_predict_with_model


class TimesfmAdapter(CallableAdapter):
    def __init__(self,
                 _model_name,
                 _enable_covariates=False,
                 _fusion_mode='early',
                 **kwargs
                 ):
        super().__init__()
        try:
            from timesfm.timesfm_torch import TimesFmTorch
            import timesfm
        except ImportError:
            raise ImportError('try to install timesfm via : `pip install timesfm[torch]`')
        if _model_name == "timesfm2":
            model_path = os.path.join(MODEL_STORAGE_PATH, 'google__timesfm-2.0-500m-pytorch', 'torch_model.ckpt')
            model_hparams = {
                "num_layers": 50,
                "use_positional_embedding": False,
                "context_len": 2048,
            }
        else:
            raise ValueError(f"{_model_name} is not supported.")
        model = TimesFmTorch(
            hparams=timesfm.TimesFmHparams(
                backend='gpu',
                per_core_batch_size=32,
                point_forecast_mode='mean',
                **model_hparams,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(path=model_path),
        )
        self.model = model
        self.fusion_mode = _fusion_mode
        self.enable_covariates = _enable_covariates

    @torch.no_grad()
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        self.model.horizon_len = _prediction_length
        if self.enable_covariates and 'context_covariates' in _input:
            fusion_mode = 'xreg + timesfm' if self.fusion_mode == 'early' else 'timesfm + xreg'
            all_covariates = {
                f'covar_{m_index}': torch.cat(
                    [_input['context_covariates'][:, m_index, :], _input['future_covariates'][:, m_index, :]],
                    dim=-1
                ).cpu()
                for m_index in range(_input['context_covariates'].shape[1])
            }
            result, _ = self.model.forecast_with_covariates(
                _input['context_target'].cpu().squeeze(1),
                dynamic_numerical_covariates=all_covariates,
                freq=[0],
                xreg_mode=fusion_mode,  # default
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,  # default
            )
            result = result[0]
        else:
            result, _ = self.model.forecast(
                _input['context_target'].cpu().squeeze(1),
                freq=[0],
            )
        mean_forecast_tensor = torch.from_numpy(result.flatten())
        return {
            'prediction': mean_forecast_tensor.to(device=_input['context_target'].device)
        }


@registry.register("timesfm2")
def timesfm2(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TimesfmAdapter(
        'timesfm2',
        _enable_covariates=False,
        **kwargs
    )
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )


@registry.register("timesfm2_xreg_early")
def timesfm2_xreg_early(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TimesfmAdapter(
        'timesfm2',
        _enable_covariates=True,
        _fusion_mode='early',
        **kwargs
    )
    return covariates_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )


@registry.register("timesfm2_xreg_late")
def timesfm2_xreg_late(_tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs) -> List[Dict]:
    adapter = TimesfmAdapter(
        'timesfm2',
        _enable_covariates=True,
        _fusion_mode='late',
        **kwargs
    )
    return covariates_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs,
    )

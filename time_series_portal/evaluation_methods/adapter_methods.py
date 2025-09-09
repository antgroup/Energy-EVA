import os
from functools import partial
from typing import List

import fev

from Core.Models.arch_adapter import MyModelAdapter
from time_series_portal.evaluation_utils.unify_predict_methods import univar_predict_with_model, \
    covariates_predict_with_model
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil
from time_series_portal.evaluation_utils.register import registry

MODEL_STORAGE_PATH = 'path_to_your_model_storage'
univar_model_name_and_path = [
    ('univar_my_model_128m', 'my_model_128m'),
]


def _to_register_univar_function(
        _model_path,
        _tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs
):
    device = kwargs.get('device', 'cuda:0')
    adapter = MyModelAdapter.load_model(_model_path).to(device)
    adapter.eval()
    return univar_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs
    )


def _to_register_covariates_function(
        _model_path,
        _tasks: List[fev.Task], _persistence_helper: PersistenceUtil, **kwargs
):
    device = kwargs.get('device', 'cuda:0')
    adapter = MyModelAdapter.load_model(_model_path).to(device)
    adapter.eval()
    return covariates_predict_with_model(
        adapter,
        _tasks,
        _persistence_helper=_persistence_helper,
        **kwargs
    )


for m_model_name, m_model_path in univar_model_name_and_path:
    m_model_path = os.path.join(MODEL_STORAGE_PATH, m_model_path)
    if not os.path.exists(m_model_path):
        raise ValueError(f'{m_model_path} is not a valid model path')
    registry.register_function(m_model_name, partial(_to_register_univar_function, m_model_path))
    registry.register_function(m_model_name + '_covariates',
                               partial(_to_register_covariates_function, m_model_path, )
                               )

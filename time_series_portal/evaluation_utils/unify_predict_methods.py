import os
from functools import partial
from typing import List, Union

import fev
from loguru import logger
from tqdm import tqdm

from Core.Models.arch_adapter.base_adapter import ArchAdapter
from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from .data_generate_util import build_univar_adapter_input_data, build_covars_adapter_input_data
from .persistence_util import PersistenceUtil

DISABLE_CACHE = os.environ.get('DISABLE_CACHE', 'false').lower() == 'true'


def univar_predict_with_model(
        _adapter: Union[CallableAdapter, ArchAdapter],
        _tasks: List[fev.Task],
        _persistence_helper: PersistenceUtil,
        **kwargs,
) -> List:
    """
    The current module is designed to abstract the implementation of single-variable inference to avoid code duplication.
    Currently, single-variable inference only utilizes the target value and time-related information.
    Users can also customize their own processing for single-variable scenarios.
    """
    device = kwargs.get("device", 'cuda:0')
    to_return_results = []

    for m_index, m_task in enumerate(_tasks, 1):
        m_task_name = os.path.basename(m_task.dataset_path)
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(_tasks)})'
        )
        if not DISABLE_CACHE:
            _persistence_helper.set_dataset(m_task)
        m_task_past_data, m_task_future_data = m_task.get_input_data()
        m_task_results = []
        m_forecast_method = partial(_adapter.generate, _prediction_length=m_task.horizon, _generate_config=dict())
        for m_past_data, m_future_data in tqdm(
                zip(m_task_past_data, m_task_future_data),
                total=len(m_task_past_data)
        ):
            m_dict_tensor, m_denormalizers = build_univar_adapter_input_data(
                [(m_past_data, m_future_data)], [m_task.dataset_info],
                device
            )
            if DISABLE_CACHE:
                # Convenient for debugging, quickly finding issues
                m_result = m_forecast_method(m_dict_tensor)
                m_prediction = m_result['prediction'].cpu().numpy().flatten()
            else:
                m_result = _persistence_helper.get_output_with_cache(m_dict_tensor, m_forecast_method)
                if m_result is not None:
                    m_prediction = m_result['prediction'].flatten()
                else:
                    continue
            m_prediction = m_denormalizers[0].inverse_transform(m_prediction)
            m_task_results.append({
                'predictions': m_prediction,
            })
        to_return_results.append(m_task_results)
        _persistence_helper.manual_flush()
    return to_return_results


def covariates_predict_with_model(
        _adapter: Union[CallableAdapter, ArchAdapter],
        _tasks: List[fev.Task],
        _persistence_helper: PersistenceUtil,
        **kwargs,
) -> List:
    """
    The current module is designed to abstract the implementation of inference with covariates to avoid code duplication.
    Currently, it includes single-variable inference that utilizes target values and time-related information,
    while additionally incorporating covariate feature information.
    Users can refer to this code to implement feature engineering, normalization,
    and other related operations in their own algorithms.
    """
    # You can implement your own algorithm to predict with covariates
    device = kwargs.get("device", 'cuda:0')
    to_return_results = []

    for m_index, m_task in enumerate(_tasks, 1):
        m_task_name = os.path.basename(m_task.dataset_path)
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(_tasks)})'
        )
        if not DISABLE_CACHE:
            _persistence_helper.set_dataset(m_task)
        m_task_past_data, m_task_future_data = m_task.get_input_data()
        m_task_results = []
        m_forecast_method = partial(_adapter.generate, _prediction_length=m_task.horizon, _generate_config=dict())
        for m_past_data, m_future_data in tqdm(
                zip(m_task_past_data, m_task_future_data),
                total=len(m_task_past_data)
        ):
            m_dict_tensor, m_denormalizers = build_covars_adapter_input_data(
                [(m_past_data, m_future_data)], [m_task.dataset_info],
                device
            )
            if DISABLE_CACHE:
                # Convenient for debugging, quickly finding issues
                m_result = m_forecast_method(m_dict_tensor)
                m_prediction = m_result['prediction'].cpu().numpy().flatten()
            else:
                m_result = _persistence_helper.get_output_with_cache(m_dict_tensor, m_forecast_method)
                if m_result is not None:
                    m_prediction = m_result['prediction'].flatten()
                else:
                    continue
            m_prediction = m_denormalizers[0].inverse_transform(m_prediction)
            m_task_results.append({
                'predictions': m_prediction,
            })
        to_return_results.append(m_task_results)
        _persistence_helper.manual_flush()
    return to_return_results

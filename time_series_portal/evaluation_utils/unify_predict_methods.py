import os
from functools import partial
from typing import Dict, List, Union

import fev
from loguru import logger
from tqdm import tqdm

from Core.Models.arch_adapter.base_adapter import ArchAdapter
from Core.Models.arch_adapter.callable_adapter import CallableAdapter
from .data_generate_util import build_univar_adapter_input_data, build_covars_adapter_input_data
from .persistence_util import PersistenceUtil

DISABLE_CACHE = os.environ.get('DISABLE_CACHE', 'false').lower() == 'true'


def _resolve_lead_gap(_task: fev.Task) -> int:
    """查询 task 需要裁掉的 past_data 末尾步数。

    用于"业务上某些步不可见"的场景(如电价 T 日预测 T+1, T 日 96 个点不可见)。
    fev 当前不支持 lead_time>1, 改在 wrapper 端裁 past_data 实现等价效果。
    值由 LeadGapTask.lead_gap 字段携带。
    """
    return int(getattr(_task, 'lead_gap', 0) or 0)


def _truncate_past_row(_past_row: Dict, _gap: int, _task: fev.Task) -> Dict:
    """把 past_row 里所有"序列型"列的末尾 _gap 步裁掉, 返回新字典。

    序列列从 task.dataset_info 推导(target/timestamp/dynamic 系列),
    其它静态列(如 item_id)原样保留。
    """
    if _gap <= 0:
        return _past_row
    info = _task.dataset_info
    seq_cols = set()
    target = info.get('target_column')
    if isinstance(target, str):
        seq_cols.add(target)
    elif target is not None:
        seq_cols.update(target)
    seq_cols.add(info.get('timestamp_column'))
    for k in ('dynamic_columns', 'known_dynamic_columns', 'past_dynamic_columns'):
        seq_cols.update(info.get(k, []) or [])
    new_row = dict(_past_row)
    for c in seq_cols:
        if c in new_row and new_row[c] is not None:
            new_row[c] = new_row[c][:-_gap]
    return new_row


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
        m_lead_gap = _resolve_lead_gap(m_task)
        m_task_results = []
        m_forecast_method = partial(_adapter.generate, _prediction_length=m_task.horizon, _generate_config=dict())
        for m_past_data, m_future_data in tqdm(
                zip(m_task_past_data, m_task_future_data),
                total=len(m_task_past_data)
        ):
            m_past_data = _truncate_past_row(m_past_data, m_lead_gap, m_task)
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
        m_lead_gap = _resolve_lead_gap(m_task)
        m_task_results = []
        m_forecast_method = partial(_adapter.generate, _prediction_length=m_task.horizon, _generate_config=dict())
        for m_past_data, m_future_data in tqdm(
                zip(m_task_past_data, m_task_future_data),
                total=len(m_task_past_data)
        ):
            m_past_data = _truncate_past_row(m_past_data, m_lead_gap, m_task)
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


def native_covar_predict_with_model(
        _adapter: Union[CallableAdapter, ArchAdapter],
        _tasks: List[fev.Task],
        _persistence_helper: PersistenceUtil,
        **kwargs,
) -> List:
    """
    Variant of `covariates_predict_with_model` for adapters that consume covariates
    natively (rather than via an external XReg fusion plugin).

    The only structural difference is that `known_dynamic_columns` is forwarded to
    the adapter via the input dict, so adapters can map covariate tensors back to
    the column names expected by upstream model APIs.
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
        m_lead_gap = _resolve_lead_gap(m_task)
        m_task_results = []
        m_forecast_method = partial(_adapter.generate, _prediction_length=m_task.horizon, _generate_config=dict())
        for m_past_data, m_future_data in tqdm(
                zip(m_task_past_data, m_task_future_data),
                total=len(m_task_past_data)
        ):
            m_past_data = _truncate_past_row(m_past_data, m_lead_gap, m_task)
            m_dict_tensor, m_denormalizers = build_covars_adapter_input_data(
                [(m_past_data, m_future_data)], [m_task.dataset_info],
                device
            )
            m_dict_tensor['known_dynamic_columns'] = m_task.dataset_info.get('known_dynamic_columns', [])
            if DISABLE_CACHE:
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

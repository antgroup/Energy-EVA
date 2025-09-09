import os
from typing import Dict, List

import fev
import numpy as np
from loguru import logger
from tqdm import tqdm

from Core.Utils.normalize_util import Normalizer, NormalizationType
from time_series_portal.evaluation_utils import registry


def predict_with_stat_model(_task: fev.Task, _model_name: str = "arima") -> List[Dict]:
    from statsforecast.models import Theta, ARIMA

    past_data, future_data = _task.get_input_data()
    if _model_name == "theta":
        model = Theta(season_length=_task.seasonality)
    elif _model_name == "arima":
        model = ARIMA()
    else:
        raise ValueError(f"Unknown model_name: {_model_name}")

    predictions = []
    normalizer = Normalizer()
    with_normalize = False
    normalization_type = NormalizationType.Z_SCORE if with_normalize else NormalizationType.NONE
    error_data_count = 0
    for m_data in tqdm(past_data, desc='predicting'):
        m_normalized_target = normalizer.fit_transform(m_data[_task.target_column], normalization_type)
        try:
            m_prediction = model.forecast(y=m_normalized_target, h=_task.horizon)["mean"].astype(np.float32)
        except Exception as e:
            m_prediction = np.ones((_task.horizon,), dtype=np.float32) * m_normalized_target[-1]
            error_data_count += 1
        predictions.append({
            "predictions": normalizer.inverse_transform(m_prediction)
        })
    if error_data_count > 0:
        logger.warning(f"There were {error_data_count} errors during prediction.")
    return predictions


@registry.register('stats_model_theta')
def stats_model_theta(_tasks: List[fev.Task], **kwargs):
    # This algorithm does not require persistence
    to_return_results = []
    for m_index, m_task in enumerate(_tasks, 1):
        m_task_name = os.path.basename(m_task.dataset_path)
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(_tasks)})'
        )
        m_task_results = predict_with_stat_model(m_task, "theta")
        to_return_results.append(m_task_results)
    return to_return_results


@registry.register('stats_model_arima')
def stats_model_arima(_tasks: List[fev.Task],**kwargs):
    # This algorithm does not require persistence
    to_return_results = []
    for m_index, m_task in enumerate(_tasks, 1):
        m_task_name = os.path.basename(m_task.dataset_path)
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(_tasks)})'
        )
        m_task_results = predict_with_stat_model(m_task, "arima")
        to_return_results.append(m_task_results)
    return to_return_results

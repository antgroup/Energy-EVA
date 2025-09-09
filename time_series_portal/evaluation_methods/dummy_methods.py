import os
from typing import List

import fev
import numpy as np
from loguru import logger

from time_series_portal.evaluation_utils.register import registry


@registry.register("dummy_model")
def dummy_model(_tasks: List[fev.Task], **kwargs):
    # dummy model directly returns the most recent historical data of the same length cycle
    # no cache needed
    to_return_results = []
    for m_index, m_task in enumerate(_tasks, 1):
        m_task_name = os.path.basename(m_task.dataset_path)
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(_tasks)})'
        )
        past_data, future_data = m_task.get_input_data()
        m_task_results = []
        for m_past_data, m_future_data in zip(past_data, future_data):
            m_task_results.append({
                'predictions': m_past_data[m_task.target_column][-m_task.horizon:].astype(np.float32),
            })
        to_return_results.append(m_task_results)
    return to_return_results

import os
from pathlib import Path
from typing import Union, Callable

import fev
from fev import Benchmark
from fev.metrics import AVAILABLE_METRICS

from Core.Utils.file_helper import glob
from time_series_portal.evaluation_utils.metric_extensions import SolarAcc, WindAcc, LoadAcc

# This section implements a hack operation by adding a new metric.
# Users can also implement their own custom metrics for evaluation.
AVAILABLE_METRICS['SOLAR_ACC'] = SolarAcc
AVAILABLE_METRICS['WIND_ACC'] = WindAcc
AVAILABLE_METRICS['LOAD_ACC'] = LoadAcc


def get_solar_tasks(_dataset_path: Union[Path, str]) -> Benchmark:
    daily_step_count = {
        'mendeley': 24,
        'pvod': 96,
        'solete': 96,
        'csg_forecast_competition': 96,
    }
    dataset_directory = os.path.join(_dataset_path, 'solar')
    to_return_benchmark = []
    for m_file_path, m_dataset_name, __ in glob(dataset_directory, {'.parquet'}):
        if m_dataset_name in daily_step_count:
            for m_max_context_length in [512, 1024, 2048]:
                to_return_benchmark.append(
                    fev.TaskGenerator(
                        dataset_path=m_file_path,
                        horizon=daily_step_count[m_dataset_name],
                        max_context_length=m_max_context_length,
                        num_rolling_windows=5,
                        rolling_step_size=daily_step_count[m_dataset_name],
                        min_context_length=daily_step_count[m_dataset_name],
                        eval_metric='SOLAR_ACC',
                        extra_metrics=['MAE', 'RMSE', 'MAPE'],
                        id_column='item_id',
                        timestamp_column='datetime',
                        target_column='avg_power',
                    )
                )

    return Benchmark(to_return_benchmark)


def get_wind_tasks(_dataset_path: Union[Path, str], _file_name_mapper: Callable = None) -> Benchmark:
    daily_step_count = {
        'europe_offshore_wind': 24,
        'mendeley': 24,
        'csg_forecast_competition': 96,
    }
    dataset_directory = os.path.join(_dataset_path, 'wind')
    to_return_benchmark = []
    # load all parquet from directory
    for m_file_path, m_dataset_name, __ in glob(dataset_directory, {'.parquet'}):
        if m_dataset_name in daily_step_count:
            for m_max_context_length in [512, 1024, 2048]:
                to_return_benchmark.append(
                    fev.TaskGenerator(
                        dataset_path=m_file_path,
                        horizon=daily_step_count[m_dataset_name],
                        max_context_length=m_max_context_length,
                        num_rolling_windows=5,
                        rolling_step_size=daily_step_count[m_dataset_name],
                        min_context_length=daily_step_count[m_dataset_name],
                        eval_metric='WIND_ACC',
                        extra_metrics=['MAE', 'RMSE', 'MAPE'],
                        id_column='item_id',
                        timestamp_column='datetime',
                        target_column='avg_power',
                    )
                )
    return Benchmark(to_return_benchmark)


def get_load_tasks(_dataset_path: Union[Path, str]) -> Benchmark:
    daily_step_count = {
        'aemo': 96,
        'entsoe': 24,
        'icsuci': 48,
        'active_power_load': 48,
    }
    dataset_directory = os.path.join(_dataset_path, 'load')
    to_return_benchmark = []
    for m_file_path, m_dataset_name, __ in glob(dataset_directory, {'.parquet'}):
        if m_dataset_name in daily_step_count:
            for m_max_context_length in [512, 1024, 2048]:
                to_return_benchmark.append(
                    fev.TaskGenerator(
                        dataset_path=m_file_path,
                        horizon=daily_step_count[m_dataset_name],
                        max_context_length=m_max_context_length,
                        num_rolling_windows=5,
                        rolling_step_size=daily_step_count[m_dataset_name],
                        min_context_length=daily_step_count[m_dataset_name],
                        eval_metric='LOAD_ACC',
                        extra_metrics=['RMSE', 'MAPE', 'MAE'],
                        id_column='item_id',
                        timestamp_column='datetime',
                        target_column='target',
                    )
                )

    return Benchmark(to_return_benchmark)

import os
from pathlib import Path
from typing import Union, Callable

import fev
import pandas as pd
import pydantic
from fev import Benchmark
from fev.metrics import AVAILABLE_METRICS

from Core.Utils.file_helper import glob
from time_series_portal.evaluation_utils.metric_extensions import SolarAcc, WindAcc, LoadAcc, PriceAcc

# This section implements a hack operation by adding a new metric.
# Users can also implement their own custom metrics for evaluation.
AVAILABLE_METRICS['SOLAR_ACC'] = SolarAcc
AVAILABLE_METRICS['WIND_ACC'] = WindAcc
AVAILABLE_METRICS['LOAD_ACC'] = LoadAcc
AVAILABLE_METRICS['PRICE_ACC'] = PriceAcc


# fev 0.5 不支持 lead_time > 1：业务上 T 日预测 T+1 且 T 日数据不可见，
# 等价做法是在 wrapper 的 build_*_adapter_input_data 之前从 past_data
# 末尾裁掉 lead_gap 个点（= T 日整天）。这里只负责把 lead_gap 透传到
# 每个窗口任务上，裁剪由消费侧完成。
@pydantic.dataclasses.dataclass
class LeadGapTask(fev.Task):
    lead_gap: int = 0


@pydantic.dataclasses.dataclass
class LeadGapTaskGenerator(fev.TaskGenerator):
    """与 fev.TaskGenerator 等价，但生成 LeadGapTask 并把 lead_gap 透传到每个窗口。"""
    lead_gap: int = 0

    def generate_tasks(self) -> list[LeadGapTask]:
        excluded = {'variants', 'num_rolling_windows', 'rolling_step_size',
                    'initial_cutoff', 'lead_gap'}
        base = {k: v for k, v in self.__dict__.items() if k not in excluded}
        tasks: list[LeadGapTask] = []
        if self.variants:
            for v in self.variants:
                tasks.append(LeadGapTask(**{**base, **v, 'lead_gap': self.lead_gap}))
        elif self.num_rolling_windows:
            for i in range(self.num_rolling_windows):
                window = dict(base)
                if isinstance(self.initial_cutoff, int):
                    window['cutoff'] = self.initial_cutoff + i * self.rolling_step_size
                else:
                    cutoff = pd.Timestamp(self.initial_cutoff)
                    if i != 0:
                        cutoff += i * pd.tseries.frequencies.to_offset(self.rolling_step_size)
                    window['cutoff'] = cutoff.isoformat()
                window['lead_gap'] = self.lead_gap
                tasks.append(LeadGapTask(**window))
        else:
            tasks.append(LeadGapTask(**base, lead_gap=self.lead_gap))
        return tasks


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


def get_price_tasks(_dataset_path: Union[Path, str]) -> Benchmark:
    # 三个 day-ahead 价格仿真集都是 15min 频率，一天 96 步。
    # 业务约束：T 日预测 T+1，T 日数据不可见。fev 当前版本不支持 lead_time>1，
    # 改用 LeadGapTask 在 wrapper 端裁掉 past_data 末尾 lead_gap 个点（= T 日整天）
    # 实现等价语义。max/min_context_length 各 +lead_gap，保证裁剪后的有效上下文
    # 仍对齐其它 scene 的 512/1024/2048。
    daily_step_count = {
        'R1_Sim': 96,
        'R2_Sim': 96,
        'R3_Sim': 96,
    }
    dataset_directory = os.path.join(_dataset_path, 'price')
    to_return_benchmark = []
    for m_file_path, m_dataset_name, __ in glob(dataset_directory, {'.parquet'}):
        if m_dataset_name not in daily_step_count:
            continue
        step = daily_step_count[m_dataset_name]
        lead_gap = step
        for m_max_context_length in [512, 1024, 2048]:
            to_return_benchmark.append(
                LeadGapTaskGenerator(
                    dataset_path=m_file_path,
                    horizon=step,
                    max_context_length=m_max_context_length + lead_gap,
                    num_rolling_windows=60,
                    rolling_step_size=step,
                    min_context_length=step + lead_gap,
                    eval_metric='PRICE_ACC',
                    extra_metrics=['MAE', 'RMSE', 'MAPE'],
                    id_column='item_id',
                    timestamp_column='datetime',
                    target_column='price',
                    lead_gap=lead_gap,
                )
            )
    return Benchmark(to_return_benchmark)

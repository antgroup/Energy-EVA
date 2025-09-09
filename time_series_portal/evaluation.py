import os.path
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, List, Dict

import datasets
import fev
import pandas as pd
from loguru import logger
from typing_extensions import LiteralString

import time_series_portal.evaluation_methods  # noqa: F401
from Core.Utils.file_helper import glob
from time_series_portal.benchmark_tasks import get_solar_tasks, get_wind_tasks, get_load_tasks
from time_series_portal.evaluation_utils import registry
from time_series_portal.evaluation_utils.logger_utils import setup_logger
from time_series_portal.evaluation_utils.persistence_util import PersistenceUtil

warnings.simplefilter("ignore")
datasets.disable_progress_bars()


def single_process_batch_benchmark(
        _model_name,
        _tasks,
        _dataset_name,
        _cache_directory,
        **kwargs,
) -> List[Dict]:
    persistence_helper = PersistenceUtil(_cache_directory, _dataset_name, _model_name, 100)
    predictions = registry.call(_model_name, _tasks, _persistence_helper=persistence_helper, _task_name=_dataset_name,
                                **kwargs)
    to_return_summaries = []
    for m_task, m_predictions in zip(_tasks, predictions):
        m_eval_summary = m_task.evaluation_summary(
            m_predictions,
            model_name=_model_name,
            trained_on_this_dataset=False
        )
        to_return_summaries.append(m_eval_summary)
    return to_return_summaries


def selected_dataset_benchmark(
        _dataset_path: Union[Path, str],
        _dataset_name: str,
        _target_path: Union[Path, str, LiteralString],
        _to_evaluate_models: List[str],
        _cache_directory: Union[Path, str],
        _log_directory: Union[Path, str],
        _concurrency: int = 1,
        **kwargs,
):
    if _dataset_name == 'solar':
        selected_tasks = get_solar_tasks(_dataset_path)
    elif _dataset_name == 'wind':
        selected_tasks = get_wind_tasks(_dataset_path)
    elif _dataset_name == 'load':
        selected_tasks = get_load_tasks(_dataset_path)
    else:
        raise ValueError(f"Unknown dataset name: {_dataset_name}")
    for m_model_index, m_model_name in enumerate(_to_evaluate_models, 1):
        m_target_result_path = Path(_target_path) / (m_model_name + ".csv")
        m_log_file = os.path.join(_log_directory, f'{m_model_name}_{_dataset_name}.log')
        setup_logger(m_log_file)
        logger.info(f"Evaluating scene [{_dataset_name}] "
                    f"with model [{m_model_name}] "
                    f"({m_model_index}/{len(_to_evaluate_models)})"
                    )
        m_summaries = single_process_batch_benchmark(
            m_model_name,
            selected_tasks.tasks,
            _dataset_name,
            _cache_directory,
            **kwargs,
        )
        m_summary_df = pd.DataFrame(m_summaries)
        m_summary_df.to_csv(m_target_result_path, index=False)


def parse_args():
    parser = ArgumentParser('Energy Dataset Benchmark')
    parser.add_argument('--scene', nargs='+', choices=['solar', 'wind', 'load'], required=True,
                        help='support task name')
    parser.add_argument('--model', nargs='+', type=str, required=True, help='model name')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--target_path', type=str, required=True, help='target path')
    parser.add_argument('--cache_path', type=str, default=None, required=False, help='cache path')
    parser.add_argument('--log_path', type=str, default=None, required=False, help='log path')
    parser.add_argument('--concurrency', type=int, default=1, required=False, help='concurrency')
    parser.add_argument('--device', type=str, default='cuda:0', required=False, help='device')
    parser.add_argument('--enable_leaderboard', action='store_true', default=False, help='enable leaderboard')
    args = parser.parse_args()
    return args


def evaluate(_args):
    to_evaluate_models = [
        m_model_name
        for m_model_name in _args.model
        if registry.has_function(m_model_name)
    ]
    assert len(to_evaluate_models) > 0, 'not select model'
    cache_path = _args.cache_path if _args.cache_path else os.path.join(_args.target_path, 'cache')
    log_path = _args.log_path if _args.log_path else os.path.join(_args.target_path, 'log')
    os.makedirs(log_path, exist_ok=True)
    for m_scene in _args.scene:
        logger.info(f"Evaluating scene [{m_scene}]")
        m_target_scene_directory = os.path.join(_args.target_path, m_scene)
        os.makedirs(m_target_scene_directory, exist_ok=True)
        selected_dataset_benchmark(
            _args.dataset_path, m_scene,
            m_target_scene_directory, to_evaluate_models,
            cache_path,
            log_path,
            _args.concurrency,
            device=_args.device,
        )
        if _args.enable_leaderboard:
            m_scene_leaderboard: pd.DataFrame = fev.leaderboard(
                summaries=[_[0] for _ in glob(m_target_scene_directory, {".csv"})],
                metric_column='MAE',
                baseline_model='dummy_model',
            )
            m_scene_leaderboard.to_csv(
                os.path.join(_args.target_path, f'{m_scene}_leaderboard.csv')
            )


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)

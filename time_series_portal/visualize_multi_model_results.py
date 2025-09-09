import io
import os
import random
import zipfile
from argparse import ArgumentParser
import hashlib

import datasets
import fev
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm

import time_series_portal.evaluation_methods  # noqa: F401
from time_series_portal.benchmark_tasks import get_load_tasks, get_wind_tasks, get_solar_tasks
from time_series_portal.evaluation_utils import registry, PersistenceUtil

datasets.disable_progress_bars()


def plot_model_comparison(_df: pd.DataFrame, _figsize=(12, 8),
                          _title='Model Comparison', _xlabel='Time Point', _ylabel='Value',
                          _historical_series=None, _historical_label='Historical Data'):
    """
    Plot multiple model results from DataFrame on the same line chart, with optional historical time series
    Parameters:
        _df: DataFrame, each column represents a model's output results, each row represents a time point
        output_file: Image file name to save
        _figsize: Image size
        _title: Image title
        _xlabel: X-axis label
        _ylabel: Y-axis label
        _historical_series: Optional, historical time series data (1D array or list)
        _historical_label: Legend label for historical data
    """

    fig, ax = plt.subplots(figsize=_figsize)
    if len(_df) > 0:
        x_future = range(0, len(_df))
        for column in _df.columns:
            ax.plot(x_future, _df[column], marker='o', linewidth=2, markersize=4, label=column)

    if _historical_series is not None and len(_historical_series) > 0:
        x_historical = range(-len(_historical_series) + 1, 1)

        ax.plot(x_historical, _historical_series,
                color='gray', linewidth=2,
                label=_historical_label, alpha=0.8)

        if len(_df) > 0:
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, ymin=0, ymax=1)

    ax.set_xlabel(_xlabel, fontsize=12)
    ax.set_ylabel(_ylabel, fontsize=12)
    ax.set_title(_title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    all_x_points = []
    if _historical_series is not None and len(_historical_series) > 0:
        all_x_points.extend(range(-len(_historical_series) + 1, 1))
    if len(_df) > 0:
        all_x_points.extend(range(1, len(_df) + 1))

    if all_x_points:
        x_min, x_max = min(all_x_points), max(all_x_points)
        major_ticks = []
        if x_max - x_min > 10:
            tick_step = max(1, (x_max - x_min) // 10)
            major_ticks = list(range(x_min, x_max + 1, tick_step))
            if x_max not in major_ticks:
                major_ticks.append(x_max)
        else:
            major_ticks = all_x_points[::max(1, len(all_x_points) // 15)]

        ax.set_xticks(major_ticks)

    if all_x_points:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    fig.tight_layout()
    return fig


def add_figure_to_zip(_zip_file, _figure, _filename, _format='png', **kwargs):
    img_buffer = io.BytesIO()

    _figure.savefig(img_buffer, format=_format, **kwargs)
    img_buffer.seek(0)

    _zip_file.writestr(_filename, img_buffer.getvalue())

    img_buffer.close()
    plt.close(_figure)


def get_context_target_and_gt_of_task(_task: fev.Task):
    gt_data = _task.get_test_data()
    target_column = _task.target_column
    to_return_gt_results = []
    to_return_context_results = []
    for m_data in gt_data:
        to_return_gt_results.append({
            'predictions': m_data[target_column]
        })
    context_data, _ = _task.get_input_data()
    for m_data in context_data:
        to_return_context_results.append(
            m_data[target_column]
        )
    return to_return_gt_results, to_return_context_results


def generate_visualization(_args):
    scene = args.scene
    tasks_limit = args.tasks_limit
    records_limit_per_task = args.records_limit
    dataset_path = args.dataset_path
    if args.cache_path is None:
        cache_path = os.path.join(args.target_path, 'cache')
    else:
        cache_path = args.cache_path
    visualization_directory = os.path.join(args.target_path, 'visualization')
    os.makedirs(visualization_directory, exist_ok=True)
    if scene == 'load':
        tasks = get_load_tasks(dataset_path).tasks
    elif scene == 'wind':
        tasks = get_wind_tasks(dataset_path).tasks
    elif scene == 'solar':
        tasks = get_solar_tasks(dataset_path).tasks
    else:
        raise NotImplementedError(f'scene {scene} not supported')

    if _args.task_shuffle and _args.specific_task is None:
        random.shuffle(tasks)
    models = sorted(list(set(_args.model)))
    candidate_methods = ['gt', ] + models
    if tasks_limit > 0 and _args.specific_task is None:
        tasks = tasks[:tasks_limit]
    record_index = 0
    records_limit = len(tasks) * records_limit_per_task
    for m_index, m_task in enumerate(tasks, 1):
        m_task_name = os.path.splitext(os.path.basename(m_task.dataset_path))[0]
        if _args.specific_task is not None:
            if m_task_name not in _args.specific_task:
                continue
        m_models_name = ','.join(models)
        # keep the file name in fixed length
        m_models_hash = hashlib.sha256(m_models_name.encode()).hexdigest()
        m_target_zip_file_path = os.path.join(
            visualization_directory,
            f'{scene}_{m_task_name}_{m_index}_{m_models_hash}.zip'
        )
        logger.info(
            f'Start evaluate dataset [{m_task_name}]({m_index}/{len(tasks)})'
        )
        m_zipfile_buffer = io.BytesIO()
        with zipfile.ZipFile(m_zipfile_buffer, 'w', zipfile.ZIP_DEFLATED) as m_target_zip_file:
            m_all_methods_results = []
            m_context_data = []
            for m_method in candidate_methods:
                if m_method == 'gt':
                    m_gt_data, m_context_data = get_context_target_and_gt_of_task(m_task)
                    m_all_methods_results.append(m_gt_data)
                    continue
                if registry.has_function(m_method):
                    m_persistence_helper = PersistenceUtil(cache_path, scene, m_method, 100)
                    m_task_results = registry.call(m_method, [m_task, ],
                                                   _persistence_helper=m_persistence_helper,
                                                   _task_name=scene,
                                                   )
                    m_all_methods_results.append(m_task_results[0])
                else:
                    logger.warning(f'{m_method} not found,skip it')
            total_records = len(m_context_data) if records_limit_per_task < 0 else records_limit_per_task
            for m_task_record_index, m_all_method_result in enumerate(
                    tqdm(zip(*m_all_methods_results), desc='process all records', total=total_records)
            ):
                if 0 < records_limit <= record_index or 0 < records_limit_per_task <= m_task_record_index:
                    break
                m_to_build_dataframe_dict = dict()
                for m_method_index, m_method_result in enumerate(m_all_method_result):
                    m_method_name = candidate_methods[m_method_index]
                    m_to_build_dataframe_dict[m_method_name] = m_method_result['predictions']
                m_record_model_comparison_dataframe = pd.DataFrame.from_dict(m_to_build_dataframe_dict)
                m_horizon_length = len(m_record_model_comparison_dataframe)
                m_record_image_name = f'{scene}_{m_task_name}_{record_index}.png'
                m_figure = plot_model_comparison(
                    m_record_model_comparison_dataframe,
                    _historical_series=m_context_data[m_task_record_index][-m_horizon_length:],
                    _figsize=(24, 8)
                )
                add_figure_to_zip(m_target_zip_file, m_figure, m_record_image_name, 'png')
                record_index += 1
        m_zipfile_buffer.seek(0)
        with open(m_target_zip_file_path, mode='wb') as to_write:
            to_write.write(m_zipfile_buffer.getvalue())
        logger.info(f'Save target zip file to {m_target_zip_file_path}')


def parse_args():
    parser = ArgumentParser('multi model result comparison visualization')
    parser.add_argument('--scene', choices=['load', 'solar', 'wind'], required=True, type=str,
                        help='scene to visualize')
    parser.add_argument('--task_shuffle', action='store_true', default=False, help='shuffle tasks')
    parser.add_argument('--model', nargs='+', type=str, required=True, help='model to visualize')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--target_path', type=str, required=True, help='target path')
    parser.add_argument('--cache_path', type=str, help='cache path')
    parser.add_argument('--tasks_limit', type=int, default=-1, help='number of tasks to visualize')
    parser.add_argument('--specific_task', type=str, nargs='+', help='specific task to visualize')
    parser.add_argument('--records_limit', type=int, default=-1, help='number of records to visualize')
    parser.add_argument('--device', type=str, default="cuda:0", help='evaluate device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    generate_visualization(args)

# Currently, fev's leaderboard is relatively coarse-grained, with datasets mixed together for calculation
# This script is designed for quick and more granular evaluation
import os.path
from argparse import ArgumentParser
from collections import defaultdict

import fev
import pandas as pd

from Core.Utils.file_helper import glob

def get_summary(
        _evaluations,
        _scene,
        _target_path,
):
    scene_overview_summary = fev.leaderboard(
        _evaluations,
        metric_column='MAE',
        baseline_model='dummy_model',
    )
    scene_overview_summary = scene_overview_summary[['gmean_relative_error', 'avg_rank']].reset_index()
    scene_all_model_avg_score = pd.concat(_evaluations).reset_index() \
        .groupby(by='model_name')[f'{_scene.upper()}_ACC'].mean()
    merged_summary = scene_overview_summary.merge(
        scene_all_model_avg_score.rename('avg_acc'),  # 重命名列名
        on='model_name',
        how='left'
    )
    merged_summary.sort_values('avg_acc', ascending=False, inplace=True)
    merged_summary.to_csv(_target_path, index=False)

def leaderboard_generate(_args):
    scenes = ['load', 'solar', 'wind']
    source_directory = _args.source_path
    target_directory = _args.target_path
    os.makedirs(target_directory, exist_ok=True)
    for m_scene in scenes:
        m_scene_directory = os.path.join(source_directory, m_scene)
        if not os.path.exists(m_scene_directory):
            continue
        m_all_models_evaluations = [
            pd.read_csv(m_file_path)
            for m_file_path, _, __ in glob(m_scene_directory, {'.csv'})
        ]
        if len(m_all_models_evaluations) == 0:
            continue
        m_scene_sub_evaluations = defaultdict(list)
        for m_model_evaluation in m_all_models_evaluations:
            for m_label, m_group_data in m_model_evaluation.groupby(by=_args.select_column):
                if _args.select_column == 'dataset_path':
                    m_label = os.path.splitext(os.path.basename(str(m_label)))[0]
                m_scene_sub_evaluations[m_label].append(m_group_data)
        for m_label, m_group_evaluations in m_scene_sub_evaluations.items():
            m_target_path = os.path.join(
                target_directory,
                f'{m_scene}_{_args.select_column}_{m_label}.csv'
            )
            get_summary(m_group_evaluations, m_scene, m_target_path)
        get_summary(m_all_models_evaluations, m_scene, os.path.join(target_directory, f'{m_scene}_overview.csv'))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--target_path", type=str, required=True, help='target directory')
    parser.add_argument("--source_path", type=str, required=True, help='source directory')
    parser.add_argument("--select_column", type=str, choices=[
        'max_context_length', 'horizon', 'dataset_path'
    ], required=True, help='select column to deep')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    leaderboard_generate(args)

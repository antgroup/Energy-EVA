from typing import Union

import datasets
import numpy as np
import pandas as pd
from fev.constants import PREDICTIONS
from fev.metrics import Metric


class LoadAcc(Metric):
    # Calculate the score for each time point of each instance, aggregate to the average daily score, and then aggregate the scores across all instances
    # Score definition
    # acc = mean_instance(
    #               1-min(rmse(gt,pred)/(max(gt)-min(gt))),1)
    #       )
    def compute(
            self,
            *,
            test_data: datasets.Dataset,
            predictions: datasets.Dataset,
            past_data: datasets.Dataset,
            seasonality: int = 1,
            quantile_levels: list[float] | None = None,
            target_column: str = "target",
            return_detail=False,
    ) -> Union[float, np.ndarray]:
        instance_count = test_data.num_rows
        prediction_data = predictions[PREDICTIONS]
        target_data = test_data[target_column]
        datetime_data = test_data['datetime']
        all_instance_score = []
        for m_instance_index in range(instance_count):
            m_to_evaluate_df = pd.DataFrame({
                'datetime': datetime_data[m_instance_index],
                'prediction': prediction_data[m_instance_index],
                'target': target_data[m_instance_index],
            })
            m_all_day_acc = []
            for m_day, m_day_df in m_to_evaluate_df.groupby(
                    by=[m_to_evaluate_df['datetime'].dt.month, m_to_evaluate_df['datetime'].dt.day]
            ):
                m_daily_count = len(m_day_df)
                m_rmse_diff = np.sqrt(np.sum(np.square(m_day_df['prediction'] - m_day_df['target'])) / m_daily_count)
                m_daily_capacity = m_day_df['target'].max() - m_day_df['target'].min()
                m_all_day_acc.append(1 - np.minimum(m_rmse_diff / m_daily_capacity, 1))
            all_instance_score.append(np.nanmean(m_all_day_acc))
        if return_detail:
            return np.asarray(all_instance_score)
        return np.nanmean(all_instance_score)


class SolarAcc(Metric):
    # Calculate the score for each time point of each instance, aggregate to the average daily score, and then aggregate the scores across all instances
    # Score definition
    # acc = mean_instance(
    #               root_mean_day(
    #                   square((gt_i-pred_i)/gt_i)
    #                       if gt_i > 0.2
    #                           else
    #                   square((gt_i-pred_i)/0.2)
    #               )
    #       )
    def compute(
            self,
            *,
            test_data: datasets.Dataset,
            predictions: datasets.Dataset,
            past_data: datasets.Dataset,
            seasonality: int = 1,
            quantile_levels: list[float] | None = None,
            target_column: str = "target",
            return_detail=False,
    ) -> Union[float, np.ndarray]:
        instance_count = test_data.num_rows
        prediction_data = predictions[PREDICTIONS]
        target_data = test_data[target_column]
        datetime_data = test_data['datetime']
        all_instance_score = []
        for m_instance_index in range(instance_count):
            m_to_evaluate_df = pd.DataFrame({
                'datetime': datetime_data[m_instance_index],
                'prediction': prediction_data[m_instance_index],
                'target': target_data[m_instance_index],
            })
            m_all_day_acc = []
            for m_day, m_day_df in m_to_evaluate_df.groupby(
                    by=[m_to_evaluate_df['datetime'].dt.month, m_to_evaluate_df['datetime'].dt.day]
            ):
                m_diff = m_day_df['prediction'] - m_day_df['target']
                m_capacity_lower_mask = m_day_df['target'] < 0.2
                m_day_df.loc[m_capacity_lower_mask, 'error'] = np.square(m_diff.loc[m_capacity_lower_mask] / 0.2)
                m_day_df.loc[~m_capacity_lower_mask, 'error'] = np.square(
                    m_diff.loc[~m_capacity_lower_mask] / m_day_df.loc[~m_capacity_lower_mask, 'target']
                )
                m_count = len(m_diff)
                m_all_day_acc.append(1 - (np.sqrt(np.sum(m_day_df['error']) / m_count)))
            all_instance_score.append(np.nanmean(m_all_day_acc))
        if return_detail:
            return np.asarray(all_instance_score)
        return np.nanmean(all_instance_score)


class WindAcc(Metric):
    # Calculate the score for each time point of each instance, aggregate to the average daily score, and then aggregate the scores across all instances
    # Score definition
    # acc = mean_instance(
    #               root_mean_day(
    #                   square((gt_i-pred_i)/gt_i)
    #                       if gt_i > 0.2
    #                           else
    #                   square((gt_i-pred_i)/0.2)
    #               )
    #       )
    def compute(
            self,
            *,
            test_data: datasets.Dataset,
            predictions: datasets.Dataset,
            past_data: datasets.Dataset,
            seasonality: int = 1,
            quantile_levels: list[float] | None = None,
            target_column: str = "target",
            return_detail=False,
    ) -> Union[float, np.ndarray]:
        instance_count = test_data.num_rows
        prediction_data = predictions[PREDICTIONS]
        target_data = test_data[target_column]
        datetime_data = test_data['datetime']
        all_instance_score = []
        for m_instance_index in range(instance_count):
            m_to_evaluate_df = pd.DataFrame({
                'datetime': datetime_data[m_instance_index],
                'prediction': prediction_data[m_instance_index],
                'target': target_data[m_instance_index],
            })
            m_all_day_acc = []
            for m_day, m_day_df in m_to_evaluate_df.groupby(
                    by=[m_to_evaluate_df['datetime'].dt.month, m_to_evaluate_df['datetime'].dt.day]
            ):
                # Avoid negative accuracy
                m_diff = np.clip(m_day_df['prediction'],0,1) - m_day_df['target']
                m_capacity_lower_mask = m_day_df['target'] < 0.2
                m_day_df.loc[m_capacity_lower_mask, 'error'] = np.square(m_diff.loc[m_capacity_lower_mask] / 0.2)
                m_day_df.loc[~m_capacity_lower_mask, 'error'] = np.square(
                    m_diff.loc[~m_capacity_lower_mask] / m_day_df.loc[~m_capacity_lower_mask, 'target']
                )
                m_count = len(m_diff)
                m_all_day_acc.append(1 - (np.sqrt(np.sum(m_day_df['error']) / m_count)))
            all_instance_score.append(np.nanmean(m_all_day_acc))
        if return_detail:
            return np.asarray(all_instance_score)
        return np.nanmean(all_instance_score)

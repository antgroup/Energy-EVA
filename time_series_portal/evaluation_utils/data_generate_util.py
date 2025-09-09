from typing import List, Union, Dict, Tuple

import numpy as np
import torch

from Core.Utils.normalize_util import Normalizer, NormalizationType


def build_univar_adapter_input_data(
        _batch_data: List,
        _batch_dataset_info: List,
        _device: Union[torch.device, str],
) -> Tuple[Dict, List[Normalizer]]:
    context_target = []
    target_normalizers = []
    for (m_past_data, _), m_dataset_info in zip(_batch_data, _batch_dataset_info):
        m_normalizer = Normalizer()
        # perform min-max scaling on target data according to context
        m_normalized_target = m_normalizer.fit_transform(
            m_past_data[m_dataset_info['target_column']], NormalizationType.MIN_MAX
        ).astype(np.float32)
        context_target.append(torch.from_numpy(m_normalized_target))
        # normalizer for inverse normalize
        target_normalizers.append(m_normalizer)
    context_target_tensor = torch.stack(context_target).to(_device)
    to_return_dict = {
        'context_target': context_target_tensor.unsqueeze(1),  # B 1 T_context,B=1
    }
    return to_return_dict, target_normalizers


def build_covars_adapter_input_data(
        _batch_data: List,
        _batch_dataset_info: List,
        _device: Union[torch.device, str],
) -> Tuple[Dict, List[Normalizer]]:
    context_target = []
    target_normalizers = []
    for (m_past_data, _), m_dataset_info in zip(_batch_data, _batch_dataset_info):
        m_normalizer = Normalizer()
        # perform min-max scaling on target data according to context
        m_normalized_target = m_normalizer.fit_transform(
            m_past_data[m_dataset_info['target_column']], NormalizationType.MIN_MAX
        ).astype(np.float32)
        context_target.append(torch.from_numpy(m_normalized_target))
        target_normalizers.append(m_normalizer)
    context_target_tensor = torch.stack(context_target).to(_device)
    future_covariates = []
    context_covariates = []

    for (m_past_data, m_future_data), m_dataset_info in zip(_batch_data, _batch_dataset_info):
        m_context_covariates = []
        m_future_covariates = []
        for m_dynamic_column in m_dataset_info['known_dynamic_columns']:
            m_normalizer = Normalizer()
            # perform min-max scaling on real value covariates data according to context
            m_normalizer.fit(m_past_data[m_dynamic_column], NormalizationType.MIN_MAX)
            m_context_normalized_data = m_normalizer.transform(
                m_past_data[m_dynamic_column],
            ).astype(np.float32)
            m_future_normalized_data = m_normalizer.transform(
                m_future_data[m_dynamic_column],
            ).astype(np.float32)
            m_context_covariates.append(torch.from_numpy(m_context_normalized_data))
            m_future_covariates.append(torch.from_numpy(m_future_normalized_data))
        m_context_covariates_tensor = torch.stack(m_context_covariates)
        m_future_covariates_tensor = torch.stack(m_future_covariates)
        context_covariates.append(m_context_covariates_tensor)
        future_covariates.append(m_future_covariates_tensor)
    context_covariates_tensor = torch.stack(context_covariates).to(_device)
    future_covariates_tensor = torch.stack(future_covariates).to(_device)
    to_return_dict = {
        'context_target': context_target_tensor.unsqueeze(1),       # B 1 T_context,B=1
        'context_covariates': context_covariates_tensor,            # B D T_context,B=1
        'future_covariates': future_covariates_tensor,              # B D T_future,B=1
    }
    return to_return_dict, target_normalizers

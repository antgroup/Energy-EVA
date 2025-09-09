from enum import Enum
from typing import Union, List, Optional

import numpy as np


class NormalizationType(Enum):
    """Normalization type enumeration"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    MAX_ABS = "max_abs"
    ROBUST = "robust"
    NONE = "none"


class Normalizer:
    """
    Normalization utility class, supporting multiple normalization methods
    sklearn-like functions

    Supported normalization methods:
    - Min-Max normalization: (x - min) / (max - min)
    - Z-Score standardization: (x - mean) / std
    - MaxAbs normalization: x / max(|x|)
    - Robust normalization: (x - median) / IQR
    """

    def __init__(self):
        self._params = {}
        self._fitted = False
        self._norm_type = None

    def fit(self,
            data: Union[np.ndarray, List],
            norm_type: Union[str, NormalizationType] = NormalizationType.MIN_MAX,
            feature_wise: bool = True) -> 'Normalizer':
        """
        Fit the data and calculate normalization parameters
        Args:
            data: Input data
            norm_type: Normalization type
            feature_wise: Whether to normalize separately by feature dimension (effective for 2D data)
        Returns:
            Normalizer: Returns itself, supporting method chaining
        """
        # convert data type
        data = np.array(data, dtype=np.float64)

        if isinstance(norm_type, str):
            norm_type = NormalizationType(norm_type.lower())
        self._norm_type = norm_type

        if norm_type == NormalizationType.MIN_MAX:
            self._fit_min_max(data, feature_wise)
        elif norm_type == NormalizationType.Z_SCORE:
            self._fit_z_score(data, feature_wise)
        elif norm_type == NormalizationType.MAX_ABS:
            self._fit_max_abs(data, feature_wise)
        elif norm_type == NormalizationType.ROBUST:
            self._fit_robust(data, feature_wise)
        elif norm_type == NormalizationType.NONE:
            pass
        else:
            raise NotImplementedError(f"not support: {norm_type}")

        self._fitted = True
        return self

    # compute parameters for min_max norm
    def _fit_min_max(self, data: np.ndarray, feature_wise: bool):
        if data.ndim == 1 or not feature_wise:
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                max_val = min_val + 1e-8
            self._params = {'min': min_val, 'max': max_val}
        else:
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            max_vals = np.where(max_vals == min_vals, min_vals + 1e-8, max_vals)
            self._params = {'min': min_vals, 'max': max_vals}

    # compute parameters for z-score norm
    def _fit_z_score(self, data: np.ndarray, feature_wise: bool):
        if data.ndim == 1 or not feature_wise:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                std_val = 1e-8
            self._params = {'mean': mean_val, 'std': std_val}
        else:
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            std_vals = np.where(std_vals == 0, 1e-8, std_vals)
            self._params = {'mean': mean_vals, 'std': std_vals}

    # compute parameters for max_abs norm
    def _fit_max_abs(self, data: np.ndarray, feature_wise: bool):
        if data.ndim == 1 or not feature_wise:
            max_abs_val = np.max(np.abs(data))
            if max_abs_val == 0:
                max_abs_val = 1e-8
            self._params = {'max_abs': max_abs_val}
        else:
            max_abs_vals = np.max(np.abs(data), axis=0)
            max_abs_vals = np.where(max_abs_vals == 0, 1e-8, max_abs_vals)
            self._params = {'max_abs': max_abs_vals}

    # compute parameters for robust(iqr) norm
    def _fit_robust(self, data: np.ndarray, feature_wise: bool):
        if data.ndim == 1 or not feature_wise:
            median_val = np.median(data)
            q75 = np.percentile(data, 75)
            q25 = np.percentile(data, 25)
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1e-8
            self._params = {'median': median_val, 'iqr': iqr}
        else:
            median_vals = np.median(data, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            iqr_vals = q75 - q25
            iqr_vals = np.where(iqr_vals == 0, 1e-8, iqr_vals)
            self._params = {'median': median_vals, 'iqr': iqr_vals}

    # sklearn-like style data transform
    def transform(self, data: Union[np.ndarray, List]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Please call the fit method first to fit the data")

        data = np.array(data, dtype=np.float64)

        if self._norm_type == NormalizationType.MIN_MAX:
            return (data - self._params['min']) / (self._params['max'] - self._params['min'])
        elif self._norm_type == NormalizationType.Z_SCORE:
            return (data - self._params['mean']) / self._params['std']
        elif self._norm_type == NormalizationType.MAX_ABS:
            return data / self._params['max_abs']
        elif self._norm_type == NormalizationType.ROBUST:
            return (data - self._params['median']) / self._params['iqr']
        elif self._norm_type == NormalizationType.NONE:
            return data

    # sklearn-like style data inverse transform
    def inverse_transform(self, data: Union[np.ndarray, List]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Please call the fit method first to fit the data")

        data = np.array(data, dtype=np.float64)

        if self._norm_type == NormalizationType.MIN_MAX:
            return data * (self._params['max'] - self._params['min']) + self._params['min']
        elif self._norm_type == NormalizationType.Z_SCORE:
            return data * self._params['std'] + self._params['mean']
        elif self._norm_type == NormalizationType.MAX_ABS:
            return data * self._params['max_abs']
        elif self._norm_type == NormalizationType.ROBUST:
            return data * self._params['iqr'] + self._params['median']
        elif self._norm_type == NormalizationType.NONE:
            return data

    def fit_transform(self,
                      data: Union[np.ndarray, List],
                      norm_type: Union[str, NormalizationType] = NormalizationType.MIN_MAX,
                      feature_wise: bool = True) -> np.ndarray:
        return self.fit(data, norm_type, feature_wise).transform(data)

    def get_params(self) -> dict:
        return self._params.copy()

    def set_params(self, params: dict, norm_type: Union[str, NormalizationType]) -> 'Normalizer':
        if isinstance(norm_type, str):
            norm_type = NormalizationType(norm_type.lower())

        self._params = params.copy()
        self._norm_type = norm_type
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def norm_type(self) -> Optional[NormalizationType]:
        return self._norm_type

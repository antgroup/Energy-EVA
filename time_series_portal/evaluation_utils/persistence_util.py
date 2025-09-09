import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Union, Callable
import fev
import h5py
import numpy as np
import torch
from loguru import logger


def convert_tensor_for_hdf5(_to_convert_object):
    """Convert tensor to numpy array for HDF5 storage"""
    if isinstance(_to_convert_object, dict):
        return {key: convert_tensor_for_hdf5(value) for key, value in _to_convert_object.items()}
    elif isinstance(_to_convert_object, (list, tuple)):
        return [convert_tensor_for_hdf5(item) for item in _to_convert_object]
    elif isinstance(_to_convert_object, np.ndarray):
        if type(_to_convert_object.dtype) == np.dtypes.DateTime64DType:
            return _to_convert_object.astype(np.int64)
        return _to_convert_object
    elif isinstance(_to_convert_object, torch.Tensor):
        return _to_convert_object.detach().cpu().numpy()
    else:
        return _to_convert_object


def hash_object(_obj: Any, _algorithm: str = 'sha256') -> str:
    hasher = hashlib.new(_algorithm)

    def _hash_recursive(item):
        if isinstance(item, dict):
            for key in sorted(item.keys()):
                _hash_recursive(key)
                _hash_recursive(item[key])
        elif isinstance(item, (list, tuple)):
            _hash_recursive(type(item).__name__)
            for element in item:
                _hash_recursive(element)
        elif isinstance(item, np.ndarray):
            _hash_recursive('numpy.ndarray')
            _hash_recursive(item.shape)
            _hash_recursive(item.dtype.str)
            if item.size > 10000:
                _hash_recursive(np.mean(item) if np.isfinite(np.mean(item)) else 0)
                _hash_recursive(np.std(item) if np.isfinite(np.std(item)) else 0)
                _hash_recursive(np.min(item) if np.isfinite(np.min(item)) else 0)
                _hash_recursive(np.max(item) if np.isfinite(np.max(item)) else 0)
            else:
                _hash_recursive(item.tobytes())
        elif isinstance(item, torch.Tensor):
            _hash_recursive('torch.Tensor')
            _hash_recursive(tuple(item.shape))
            _hash_recursive(str(item.dtype))
            _hash_recursive(str(item.device))
            if item.numel() > 10000:
                _hash_recursive(float(torch.mean(item).item()) if torch.isfinite(torch.mean(item)) else 0)
                _hash_recursive(float(torch.std(item).item()) if torch.isfinite(torch.std(item)) else 0)
                _hash_recursive(float(torch.min(item).item()) if torch.isfinite(torch.min(item)) else 0)
                _hash_recursive(float(torch.max(item).item()) if torch.isfinite(torch.max(item)) else 0)
            else:
                _hash_recursive(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, (int, float, str, bool, type(None))):
            hasher.update(str(item).encode('utf-8'))
        elif isinstance(item, bytes):
            hasher.update(item)
        else:
            _hash_recursive(type(item).__name__)
            try:
                hasher.update(str(item).encode('utf-8'))
            except Exception:
                hasher.update(str(id(item)).encode('utf-8'))

    _hash_recursive(_obj)
    return hasher.hexdigest()


def save_pickle_to_hdf5_group(_group, _dataset_name: str, _obj: Any):
    """Serialize object using pickle and save to HDF5 group"""
    pickled_data = pickle.dumps(_obj)
    dt = h5py.vlen_dtype(np.dtype('uint8'))
    if _dataset_name in _group:
        del _group[_dataset_name]
    dataset = _group.create_dataset(_dataset_name, (1,), dtype=dt)
    dataset[0] = np.frombuffer(pickled_data, dtype=np.uint8)


def load_pickle_from_hdf5_group(_group, _dataset_name: str) -> Any:
    """Load pickled object from HDF5 group"""
    # Read data from variable-length array
    pickled_array = _group[_dataset_name][0]  # Get first element
    # Convert to bytes
    if isinstance(pickled_array, np.ndarray):
        pickled_bytes = pickled_array.tobytes()
    else:
        # If it's a special h5py array type
        pickled_bytes = bytes(pickled_array)
    # Deserialize
    return pickle.loads(pickled_bytes)


class PersistenceUtil:
    """
    The current tool is designed to avoid the time consumption caused by repeated model inference by persisting historical results as h5 files.
    As long as the data comes from the same dataset and the input data is identical,
    the results can be quickly obtained through the persisted cache,
    reducing unnecessary time and resource consumption.
    If the results are not in the cache, the model will be called for inference.
    """
    def __init__(self, _cache_directory: Union[Path, str], _scene_name: str, _method_name: str,
                 _max_cache_size: int = 20):
        cache_directory = Path(_cache_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
        self.cache_directory = cache_directory
        self.method_name = _method_name
        self.h5_file_path = self.cache_directory / f"{_method_name}_{_scene_name}.h5"
        self.scene_name = _scene_name

        # Memory cache
        self.input_cache = {}
        self.output_cache = {}

        # Existing disk records
        self.existing_outputs = dict()

        # Configuration
        self.max_cache_size = _max_cache_size

    def set_cache_size(self, _cache_size):
        self.max_cache_size = _cache_size

    def set_dataset(self, _task: fev.Task):
        """Set dataset without creating separate files"""
        self.manual_flush()
        task_info = _task.to_dict()
        # Optimize groupname logic to avoid cache invalidation when data path changes
        data_path = task_info['dataset_path']
        # If filename has extension, use hyphen as separator
        # Assume filenames like:
        # /path/to/dataset/pvod_solar-0.4.parquet
        # /path/to/dataset/pvod_solar.parquet
        task_info['dataset_path'] = '%s_%s' % (
            self.scene_name,
            os.path.splitext(os.path.basename(data_path))[0].rsplit('-', 1)[0]
        )
        task_fingerprint = hash_object(task_info)

        # Use task fingerprint as group name instead of creating new file
        self.current_task_group = task_fingerprint
        self._ensure_h5_file_structure()
        self._preload_existing_outputs()

    def _ensure_h5_file_structure(self):
        """Ensure HDF5 file structure exists"""
        if not self.h5_file_path.exists():
            with h5py.File(self.h5_file_path, 'w') as f:
                pass  # Create empty file

        # Ensure current task group exists
        task_group_exist = True
        with h5py.File(self.h5_file_path, 'r', swmr=True) as f:
            if self.current_task_group not in f:
                task_group_exist = False
        if not task_group_exist:
            # Ensure only one process edits current file
            with h5py.File(self.h5_file_path, 'a') as f:
                task_group = f.create_group(self.current_task_group)
                task_group.create_group('inputs')
                task_group.create_group('outputs')

    def _preload_existing_outputs(self):
        """Preload existing outputs"""
        if not self.h5_file_path.exists():
            return

        try:
            with h5py.File(self.h5_file_path, 'r', swmr=True) as f:
                if self.current_task_group in f and 'outputs' in f[self.current_task_group]:
                    logger.info('Start loading cache')
                    outputs_group = f[self.current_task_group]['outputs']
                    self.existing_outputs = self._read_pickle_dict_from_hdf5_group(outputs_group)
                    logger.info(f'Finished loading cache with {len(self.existing_outputs)} items')
        except Exception as e:
            logger.opt(exception=True).warning(f"Loading cache failed: {e}")

    def _read_pickle_dict_from_hdf5_group(self, _group) -> Dict:
        """Read pickled dictionary data from HDF5 group"""
        result = {}

        # Only read datasets (pickle serialized data)
        for key in _group.keys():
            if isinstance(_group[key], h5py.Dataset):
                # Check if it's variable-length array (pickle storage format)
                try:
                    # Try to load pickled data
                    result[key] = load_pickle_from_hdf5_group(_group, key)
                except Exception as e:
                    # If not pickle format, try normal reading
                    try:
                        result[key] = _group[key][()]
                    except:
                        logger.warning(f"Failed to load dataset {key}")
                        result[key] = None
            elif isinstance(_group[key], h5py.Group):
                result[key] = self._read_pickle_dict_from_hdf5_group(_group[key])
        return result

    def _flush_to_disk(self):
        """Flush cache to disk"""
        if not hasattr(self, 'current_task_group'):
            return

        if not self.input_cache and not self.output_cache:
            return

        try:
            self._ensure_h5_file_structure()
            # Ensure only one process edits current file
            with h5py.File(self.h5_file_path, 'a') as to_write:
                task_group = to_write[self.current_task_group]

                # Write inputs
                if self.input_cache:
                    inputs_group = task_group['inputs']
                    for m_hash_key, m_data in self.input_cache.items():
                        try:
                            save_pickle_to_hdf5_group(inputs_group, m_hash_key, m_data)
                        except Exception as e:
                            logger.warning(f"Failed to save input {m_hash_key}: {e}")
                    self.input_cache.clear()

                # Write outputs
                if self.output_cache:
                    outputs_group = task_group['outputs']
                    for m_hash_key, m_data in self.output_cache.items():
                        try:
                            save_pickle_to_hdf5_group(outputs_group, m_hash_key, m_data)
                        except Exception as e:
                            logger.warning(f"Failed to save output {m_hash_key}: {e}")
                    self.output_cache.clear()

        except Exception as e:
            logger.opt(exception=True).warning(f"Failed to flush cache to disk: {e}")

    def set_input(self, _input_data: Union[Dict], _sink=False) -> str:
        """Set input data"""
        input_hash = hash_object(_input_data)
        if _sink:
            self.input_cache[input_hash] = convert_tensor_for_hdf5(_input_data)
        return input_hash

    def set_output(self, _callback_id: str, _output_data: Union[Dict]):
        """Set output data"""
        converted_data = convert_tensor_for_hdf5(_output_data)
        self.output_cache[_callback_id] = converted_data
        self.existing_outputs[_callback_id] = converted_data

        # Flush when exceeding cache size
        if len(self.output_cache) >= self.max_cache_size:
            self._flush_to_disk()

    def get_output(self, _callback_id: str) -> Union[Dict, None]:
        """Get output data"""
        if _callback_id in self.existing_outputs:
            return self.existing_outputs[_callback_id]
        return None

    def get_output_with_cache(self, _input_data: Union[Dict], _to_request_method: Callable) -> Union[Dict, None]:
        """Get output with cache"""
        callback_id = self.set_input(_input_data, _sink=False)
        try:
            exist_output = self.get_output(callback_id)
            if exist_output is None:
                to_sink_result = _to_request_method(_input_data)
                self.set_output(callback_id, to_sink_result)
                to_return_result = self.get_output(callback_id)
                assert to_return_result is not None, 'sink file missing from cache'
                return to_return_result
            else:
                return exist_output
        except Exception as e:
            logger.opt(exception=True).warning(f'Computation failed with error {e}')
            return None

    def manual_flush(self):
        """Manually flush cache"""
        self._flush_to_disk()


if __name__ == '__main__':
    import fev

    helper = PersistenceUtil(
        './test_cache_directory', 'solar', 'test_method',
    )
    test_task = fev.Task(
        dataset_path='/PATH/TO/DATASET/solar/ningxia.parquet',
        horizon=96,
        eval_metric='MAE',
        extra_metrics=['RMSE', 'MAPE'],
        id_column='item_id',
        timestamp_column='datetime',
        target_column='avg_power',
    )
    helper.set_dataset(test_task)
    past_data, future_data = test_task.get_input_data()
    for m_past_data, m_future_data in zip(past_data, future_data):
        m_to_input_data = {
            'context_target': m_past_data['avg_power'],
        }
        m_output_data = {
            'predictions': m_past_data['avg_power'][..., -test_task.horizon:],
        }
        m_id = helper.set_input(m_to_input_data, False)
        helper.set_output(m_id, m_output_data)
        m_data = helper.get_output(m_id)
    helper.manual_flush()
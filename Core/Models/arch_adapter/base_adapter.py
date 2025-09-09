import json
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Union

from torch import nn
from torch.nn import Module

"""
ArchAdapterConfig is used for configuration of different models
The reason for converting configuration files to dataclass is to have code completion and corresponding type restrictions when getting config later
"""


@dataclass
class ArchAdapterConfig:
    arch_name: str = 'default_arch'
    pretrained_model: str = ''

    # build config class
    @classmethod
    def load_dict_config(cls, _config: Union[Dict, str, Path]):
        to_return_config = cls()
        if isinstance(_config, dict):
            config = _config
        elif isinstance(_config, str) or isinstance(_config, Path):
            config_path = Path(_config)
            if config_path.exists() and config_path.is_file():
                with open(_config, mode='r', encoding='utf-8') as to_read:
                    config = json.loads(to_read.read())
            else:
                raise ValueError(f'{_config} is not a valid path')
        else:
            raise NotImplementedError(f'{type(_config)} config is not supported.')
        for m_key in config.keys():
            if hasattr(to_return_config, m_key):
                setattr(to_return_config, m_key, config[m_key])
        return to_return_config

    # convert current dict to dict-like config
    def to_dict(self):
        return asdict(self)


"""
ArchAdapter is used for data adaptation of different models, and implements the structure according to specifications
"""


class ArchAdapter(nn.Module, ABC):
    config_class = ArchAdapterConfig

    def __init__(self, _config: ArchAdapterConfig, ):
        super().__init__()
        self.config = _config

    @abstractmethod
    def build_model(self) -> Module:
        # Build the model and load parameters
        pass

    @abstractmethod
    def preprocess(self, _input) -> Dict:
        # Process the data into the format required by the current model
        pass

    @abstractmethod
    def post_process(self, _input, _output) -> Dict:
        # Scale back the prediction results
        pass

    def forward(self, **_input) -> Dict:
        processed_input_tensor = self.preprocess(_input)
        cache, output = self.model(**processed_input_tensor)
        processed_output = self.post_process(_input, output)
        return dict(
            cache=cache,
            output=processed_output,
        )

    @abstractmethod
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict):
        pass

    # Load model from persisted model and configuration files
    @classmethod
    def load_model(cls, _model_directory: Union[Path, str]):
        _model_directory = Path(_model_directory)
        if not _model_directory.exists():
            raise FileNotFoundError(f'{_model_directory} is not a valid path')
        model_path = _model_directory / 'model.pt'
        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} is not a valid path')
        config_path = _model_directory / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'{config_path} is not a valid path')
        config = cls.config_class.load_dict_config(config_path)
        config.pretrained_model = str(model_path)
        model = cls(config)
        return model

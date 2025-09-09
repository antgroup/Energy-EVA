from abc import abstractmethod, ABC
from typing import Dict


# Any model, or even HTTP services, can get specified results from requests based on the current interface paradigm
class CallableAdapter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, _input: Dict, _prediction_length: int, _generate_config: Dict) -> Dict:
        """
        time series prediction generate

        :param _input:  all available known data
        :param _prediction_length:  to predict length
        :param _generate_config:    required generation configuration
        :return:    the returned dict must have a 'prediction' key that holds the predicted results for a time series sequence
        """
        pass

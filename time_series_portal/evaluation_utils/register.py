from typing import Optional, Any, Callable, Dict

from loguru import logger


class GlobalRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, _name: Optional[str] = None):
        def decorator(func: Callable) -> Callable:
            register_name = _name or func.__name__
            self._registry[register_name] = func
            return func
        logger.info(f'{_name} register finish')
        return decorator

    def register_function(self, _name, _callable_function):
        self._registry[_name] = _callable_function
        logger.info(f'{_name} register finish')

    def call(self, _name: str, *args, **kwargs) -> Any:
        func = self._registry.get(_name)
        if func is None:
            raise KeyError(f"Function '{_name}' not registered")
        logger.info(f"Start evaluate model [{_name}]")
        to_return_result = func(*args, **kwargs)
        logger.info(f"Finish evaluate model [{_name}]")
        return to_return_result

    def list_functions(self) -> list:
        return list(self._registry.keys())

    def has_function(self, _name: str) -> bool:
        return _name in self._registry


# global instance
registry = GlobalRegistry()

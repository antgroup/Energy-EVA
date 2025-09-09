# Important Notice

This package is used to test the performance of different third-party algorithms on the current dataset.

Due to the significant dependency differences among various algorithms, it is recommended to use separate virtual environments for evaluating each algorithm (`uv` is recommended here). You only need to ensure that all model evaluation results are stored in the same folder.

The algorithm environments can be retained or discarded as needed. If some algorithms do not require evaluation, you should comment out their imports in `__init__.py` to avoid "dependency not found" errors.
# Energy&Electricity Time-Series Benchmark

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
## Overview
EnergyEVA is a benchmark designed for zero-shot time series forecasting in the energy and electricity (e.g., renewable power generation and industrial electricity usage). 
It provides standardized evaluation methods and multiple datasets to assess model performance in real-world energy scenarios.
## Key Features
- Curated zero-shot time series dataset for energy and electricity applications
- Customized evaluation metrics for energy and electricity forecasting
- Comparative evaluation of open-source models: Moirai, Chronos, TiRex, Sundial, TOTO, and TimesFM

## Scene Description
We provide evaluation benchmarks for three scenarios:
1. **Univariate electricity load forecasting** - Single variable power consumption prediction (with datetime information)
2. **Photovoltaic power generation forecasting with meteorological covariates** - Solar power prediction incorporating weather data and datetime information
3. **Wind power generation forecasting with meteorological covariates** - Wind energy prediction incorporating weather data and datetime information
Each scenario contains multiple sub-datasets. All datasets are sourced from public and traceable data platforms. Through data preprocessing and processing techniques, the raw data is transformed into standardized evaluation data files.
## Data Description
| Scene | Sub Dataset Name                  | Instance Num | Timestep Num | Source                                                                                                                                                                                                               |
|-------|-----------------------------------|-------------:|-------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Solar | csg_forecast_competition          |          107 |      304,760 | [Link](https://www.nature.com/articles/s41597-022-01696-6)                                                                                                                                                           |
| Solar | mendeley                          |            8 |        5,824 | [Link](https://data.mendeley.com/datasets/gxc6j5btrx/1)                                                                                                                                                              |
| Solar | pvod                              |           59 |      142,085 | [Link](https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62)                                                                                                                                    |
| Solar | solete                            |           10 |       29,184 | [Link](https://data.dtu.dk/articles/dataset/The_SOLETE_dataset/17040767?file=40097803)                                                                                                                               |
| Wind  | csg_forecast_competition          |          157 |      293,977 | [Link](https://www.nature.com/articles/s41597-022-01696-6)                                                                                                                                                           |
| Wind  | mendeley                          |           12 |        8,752 | [Link](https://data.mendeley.com/datasets/gxc6j5btrx/1)                                                                                                                                                              |
| Wind  | europe_offshore_wind              |        1,160 |   10,168,560 | [Link](https://figshare.com/articles/dataset/Dataset_for_the_Paper_Analyzing_Europe_s_Biggest_Offshore_Wind_Farms_a_Data_set_With_40_Years_of_Hourly_Wind_Speeds_and_Electricity_Production_/19139648?file=34079588) |
| Load  | aemo                              |           40 |      117,256 | [Link](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data)                                                                                                  |
| Load  | entsoe                            |           73 |       53,352 | [Link](https://transparency.entsoe.eu/)                                                                                                                                                                              |
| Load  | active_power_load                 |            7 |        5,160 | [Link](https://data.mendeley.com/datasets/jxm8d4w4cv/1)                                                                                                                                                              |
| Load  | icsuci                            |           64 |       90,930 | [Link](https://archive.ics.uci.edu/dataset/1158/high-resolution+load+dataset+from+smart+meters+across+various+cities+in+morocco)                                                                                     |

[dataset download link](https://zenodo.org/records/17099628)

## Project Structure
```shell
├── Core # Common model modules or other utilities
│   ├── Models
│   ├── Utils
│   └── __init__.py
├── LEGAL.md
├── LICENSE
├── Readme.md # this file
├── pyproject.toml # project requirements with `uv`
├── time_series_portal # time series benchmark main entrance
│   ├── __init__.py
│   ├── benchmark_tasks.py  # load difference scene benchmark tasks
│   ├── config.py # some common config,such as model storage path
│   ├── evaluation.py # run this for evaluation
│   ├── evaluation_methods # implemented evaluation methods
│   ├── evaluation_utils  # time series evaluation utils
│   ├── leaderboard_generate.py # generate customized leaderboard
│   └── visualize_multi_model_results.py  # generate multi-models prediction visualization
└── uv.lock # generated by `uv`

```
The current framework utilizes **[fev](https://github.com/autogluon/fev)** as the evaluation framework. The system works by reading datasets and converting them into **Context** and **Future** parts to call models for inference.

All models should inherit from either `ArchAdapter` or `CallableAdapter` and implement the `generate` function. The callable algorithms are then registered to the registry via `@registry.register("algorithm_name")`.

The benchmark process is conducted through the following components:

- `evaluation.py` - Script for batch benchmarking
- `visualize_multi_model_results.py` - Visualization of results from multiple models
- `leaderboard_generate.py` - Leaderboard construction


## How to use
Create virtual environment:
```shell
# cd into EnergyEVA
pip install uv
uv venv . # you can replace this with your ideal virtual environment path
uv sync   # install basic requirements,if you need to run other third-party models, you need to install other dependent packages(e.g. chronos-forecasting)
source .venv/bin/activate
```

Run evaluation:
```shell
python time_series_portal/evaluation.py \ 
        --dataset_path PATH/TO/YOUR/DATASET/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --scene wind load solar \
        --model dummy_model \
```
you can use `python time_series_portal/evaluation.py --help` to view more configuration options

Generate visualization of multi-models:
```shell
python time_series_portal/visualize_multi_model_results.py \
        --dataset_path PATH/TO/YOUR/DATASET/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --scene load \
        --model dummy_model toto_151m
```
you can use `python time_series_portal/visualize_multi_model_results.py --help` to view more configuration options


Generate leaderboard among multi-models:
```shell
python time_series_portal/leaderboard_generate.py \
        --source_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION/leaderboard \
        --select_column dataset_path
```

## How to Develop Your Own Algorithm Benchmark
There are some examples in `time_series_portal/evaluation_methods/third_party_methods`. 
You can implement your model in `Core/Models/arch_adapter`, and call it in `time_series_portal/evaluation_methods/adapter_methods.py`.
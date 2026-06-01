# Energy Scenario Evaluation Dataset and Benchmark (Energy-EVA)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

## Updates!!
- **[2026-06-01]**: Released **Energy-EVA 2.0** — added a new day-ahead electricity price forecasting scene (3 sub-datasets); upgraded four third-party baselines (Chronos-2, Moirai-2.0-R-small, Toto-2.0-2.5B, TimesFM-2.5-200M); added EnergyTS V3.0.

## Overview
The Energy Scenario Evaluation Dataset and Benchmark (Energy-EVA) serves as a dedicated evaluation standard for applications in the energy domain, focusing currently on zero-shot time series forecasting tasks, including renewable energy production and industrial usage. It provides a consistent framework and datasets for assessing model generalization within practical energy environments, and features a versatile structure to facilitate the future incorporation of multi-modal tasks. Looking ahead, Energy-EVA plans to broaden its scope to encompass various task types, including applications for energy-oriented large language models and visual tasks for contexts such as power grid inspections.
## Key Features
- Time series datasets tailored for applications in energy and electricity
- Customized evaluation metrics for precise energy and electricity forecasting
- Performance comparison of leading open-source models (Moirai, Chronos, TiRex, Sundial, TOTO, TimesFM) alongside our proprietary model EnergyTS V3.0.

## Scene Description
We provide evaluation benchmarks for four scenarios:
1. **Univariate electricity load forecasting** - Single variable power consumption prediction with datetime information
2. **Photovoltaic power generation forecasting with meteorological covariates** - Solar power prediction incorporating weather data and datetime information
3. **Wind power generation forecast with meteorological covariates** - Wind power prediction incorporating weather data and datetime information.
4. **Day-ahead electricity price forecasting** - Hourly day-ahead electricity price prediction with datetime information.


Every scenario encompasses several sub-datasets. All data is obtained from publicly accessible and traceable platforms. Using data pre-processing and processing methods, the initial raw data is converted to standardized evaluation data files.
## Dataset Description

[Download link for Solar/Wind/Load](https://zenodo.org/records/17099628)

[Download link for Price](https://zenodo.org/records/20485935)

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
| Load  | icsuci                            |           64 |       90,930 | [Link](https://archive.ics.uci.edu/dataset/1158/high-resolution+load+dataset+from+smart+meters+across+various+cities+in+morocco)                |
| Price | R1_Sim                            |            1 |        8,640 | -                                                                                                                                                                                                                    |
| Price | R2_Sim                            |            1 |        8,832 | -                                                                                                                                                                                                                    |
| Price | R3_Sim                            |            1 |        8,640 | -                                                                                                                                                                                                                    |

## Leaderboard

Access the [evaluation detail](./evaluation_results/) to examine comprehensive information.

### Solar Power Generation Forecasting

| model_name            | gmean_relative_error | avg_rank | avg_acc |
| --------------------- | -------------------- | -------- | ------- |
| EnergyTS_V3.0         | 0.4452               | 1.90     | 0.8296  |
| chronos-2             | 0.4442               | 1.50     | 0.8274  |
| timesfm2.5_xreg_early | 0.5910               | 3.63     | 0.7938  |
| toto_2.0_2.5B         | 0.7595               | 3.62     | 0.7074  |
| tirex                 | 0.9201               | 5.70     | 0.6724  |
| moirai_2.0_R_small    | 0.9924               | 6.50     | 0.6555  |
| sundial_base_128m     | 0.9749               | 6.30     | 0.6452  |
| dummy_model           | 1.0000               | 6.85     | 0.6263  |

### Wind Power Generation Forecasting

| model_name            | gmean_relative_error | avg_rank | avg_acc |
| --------------------- | -------------------- | -------- | ------- |
| EnergyTS_V3.0         | 0.0732               | 1.31     | 0.8292  |
| chronos-2             | 0.2393               | 1.69     | 0.7263  |
| timesfm2.5_xreg_early | 0.3725               | 3.60     | 0.6134  |
| tirex                 | 0.6887               | 4.87     | 0.3619  |
| sundial_base_128m     | 0.7046               | 5.98     | 0.3617  |
| toto_2.0_2.5B         | 0.6797               | 4.60     | 0.3591  |
| moirai_2.0_R_small    | 0.7065               | 5.96     | 0.3494  |
| dummy_model           | 1.0000               | 8.00     | 0.0462  |

### Power Load Forecasting

| model_name            | gmean_relative_error | avg_rank | avg_acc |
| --------------------- | -------------------- | -------- | ------- |
| EnergyTS_V3.0         | 0.5877               | 3.47     | 0.7071  |
| toto_2.0_2.5B         | 0.6125               | 2.90     | 0.6938  |
| chronos-2             | 0.6198               | 3.03     | 0.6921  |
| moirai_2.0_R_small    | 0.6287               | 3.67     | 0.6894  |
| timesfm2.5_xreg_early | 0.7549               | 4.70     | 0.6725  |
| sundial_base_128m     | 0.7569               | 5.12     | 0.6689  |
| tirex                 | 0.7766               | 5.40     | 0.6688  |
| dummy_model           | 1.0000               | 7.72     | 0.6163  |

### Day-ahead Electricity Price Forecasting

| model_name            | gmean_relative_error | avg_rank | avg_acc |
| --------------------- | -------------------- | -------- | ------- |
| EnergyTS_V3.0         | 0.5974               | 2.64     | 0.8874  |
| chronos-2             | 0.7007               | 3.51     | 0.8699  |
| timesfm2.5_xreg_early | 0.7237               | 3.96     | 0.8478  |
| toto_2.0_2.5B         | 0.8163               | 4.39     | 0.8447  |
| tirex                 | 0.8820               | 4.98     | 0.8381  |
| moirai_2.0_R_small    | 0.9130               | 5.22     | 0.8321  |
| sundial_base_128m     | 0.9590               | 5.59     | 0.8274  |
| dummy_model           | 1.0000               | 5.71     | 0.8152  |

## Project Structure
```shell
├── Core # Modules that are commonly used or various utilities
│   ├── Models
│   ├── Utils
│   └── __init__.py
├── LEGAL.md
├── LICENSE
├── README.md # This document
├── pyproject.toml # Project requirements with `uv`
├── time_series_portal # Primary entrance for time series benchmark
│   ├── __init__.py
│   ├── benchmark_tasks.py  # Load difference scene benchmark tasks
│   ├── config.py # Several typical configurations, like the path for storing the model
│   ├── evaluation.py # Execute this for assessment
│   ├── evaluation_methods # Implemented evaluation methods
│   │   └── third_party_methods # Contains v1 baselines (chronos/, moirai/, timesfm/, toto/, tirex/, sundial/) and v2 baselines (chronos2/, moirai2/, timesfm25/, toto2/)
│   ├── evaluation_utils  # Utilities for evaluating time series
│   ├── leaderboard_generate.py # Generate customized leaderboard
│   └── visualize_multi_model_results.py  # Visualize predictions from multiple models
└── uv.lock # Generated by `uv`

```
Energy-EVA employs **[fev](https://github.com/autogluon/fev)** for time-series forecasting evaluation purposes. The system operates by interpreting datasets and transforming them into **Context** and **Future** segments to invoke models for inference.

Customized models must derive from either `ArchAdapter` or `CallableAdapter` and include the `generate` method. These callable algorithms are then registered within the registry using `@registry.register("algorithm_name")`.

The benchmarking process comprises the following components:

- `evaluation.py` - Script for batch benchmarking
- `visualize_multi_model_results.py` - Visualization of results from multiple models
- `leaderboard_generate.py` - Leaderboard construction


## Pipeline
Create virtual environment:
```shell
# cd into Energy-EVA
pip install uv
uv venv . # Substitute this with the path to your specific virtual environment
uv sync   # Install basic requirements. For third-party models, add necessary packages (e.g., chronos-forecasting).
source .venv/bin/activate
```

Run evaluation:
```shell
python time_series_portal/evaluation.py \ 
        --dataset_path PATH/TO/YOUR/DATASET/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --scene wind load solar price \
        --model dummy_model \
```
Use `python time_series_portal/evaluation.py --help` to view more configuration options

Generate visualization of multi-models:
```shell
python time_series_portal/visualize_multi_model_results.py \
        --dataset_path PATH/TO/YOUR/DATASET/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --scene load \
        --model dummy_model toto_2.0_2.5B
```
Use `python time_series_portal/visualize_multi_model_results.py --help` to view more configuration options


Generate leaderboard among multi-models:
```shell
python time_series_portal/leaderboard_generate.py \
        --source_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION/leaderboard \
        --select_column dataset_path
```

## Evaluate proprietary algorithms
Examples are available in `time_series_portal/evaluation_methods/third_party_methods`. Implement your model in `Core/Models/arch_adapter` and invoke it via `time_series_portal/evaluation_methods/adapter_methods.py`.

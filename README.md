# Energy Scenario Evaluation Dataset and Benchmark (Energy-EVA)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
## Overview
The Energy Scenario Evaluation Dataset and Benchmark (Energy-EVA) serves as a dedicated evaluation standard for applications in the energy domain, focusing currently on zero-shot time series forecasting tasks, including renewable energy production and industrial usage. It provides a consistent framework and datasets for assessing model generalization within practical energy environments, and features a versatile structure to facilitate the future incorporation of multi-modal tasks. Looking ahead, Energy-EVA plans to broaden its scope to encompass various task types, including applications for energy-oriented large language models and visual tasks for contexts such as power grid inspections.
## Key Features
- Time series datasets tailored for applications in energy and electricity
- Customized evaluation metrics for precise energy and electricity forecasting
- Performance comparison of leading open-source models (i.e., Moirai, Chronos, TiRex, Sundial, TOTO, and TimesFM, along with EnergyTS 2.0) alongside our proprietary model, EnergyTS 2.0.

## Scene Description
We provide evaluation benchmarks for three scenarios:
1. **Univariate electricity load forecasting** - Single variable power consumption prediction with datetime information
2. **Photovoltaic power generation forecasting with meteorological covariates** - Solar power prediction incorporating weather data and datetime information
3. **Wind power generation forecast with meteorological covariates** - Wind power prediction incorporating weather data and datetime information.


Every scenario encompasses several sub-datasets. All data is obtained from publicly accessible and traceable platforms. Using data pre-processing and processing methods, the initial raw data is converted to standardized evaluation data files.
## Dataset Description

[Download link](https://zenodo.org/records/17099628)

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
| Load  | icsuci                            |           64 |       90,930 | [Link](https://archive.ics.uci.edu/dataset/1158/high-resolution+load+dataset+from+smart+meters+across+various+cities+in+morocco)                

## Leaderboard

Access the [evaluation detail](./evaluation_results/) to examine comprehensive information.

### Solar Power Generation Forecasting

| model_name          | gmean_relative_error | avg_rank           | avg_acc            |
|---------------------|----------------------|--------------------|--------------------|
| EnergyTS_V2.0       | 0.4777164058883706   | 1.0666666666666667 | 0.8147761492027971 |
| timesfm2_xreg_early | 0.5871783598600845   | 2.3333333333333335 | 0.7969484925270081 |
| chronos-bolt-base   | 0.9097303727694933   | 4.016666666666667  | 0.6748293007392193 |
| tirex               | 0.9201187633451223   | 4.566666666666666  | 0.6723568061987559 |
| moirai_1.1_R_large  | 1.0478644270451873   | 6.583333333333333  | 0.6552334736747231 |
| sundial_base_128m   | 0.9748742453493596   | 5.433333333333334  | 0.6452172885338465 |
| toto_151m           | 0.9646401855164273   | 5.866666666666666  | 0.6448465446631114 |
| dummy_model         | 1.0                  | 6.133333333333334  | 0.6263204350719228 |

### Wind Power Generation Forecasting

| model_name          | gmean_relative_error | avg_rank           | avg_acc             |
|---------------------|----------------------|--------------------|---------------------|
| EnergyTS_V2.0       | 0.10611033427200027  | 1.0                | 0.78701318340354    |
| timesfm2_xreg_early | 0.4001787937577178   | 2.888888888888889  | 0.5964015142785178  |
| toto_151m           | 0.7027388019708346   | 4.333333333333333  | 0.3670413409670194  |
| tirex               | 0.6887401773801904   | 3.6444444444444444 | 0.36189989298582076 |
| sundial_base_128m   | 0.7046367148681573   | 4.888888888888889  | 0.36166646977265676 |
| moirai_1.1_R_large  | 0.7039745145049469   | 4.7555555555555555 | 0.35574313039817274 |
| chronos-bolt-base   | 0.7827271146276737   | 6.511111111111111  | 0.3151765439878566  |
| dummy_model         | 1.0                  | 7.977777777777778  | 0.04616299244696686 |

### Power Load Forecasting

| model_name          | gmean_relative_error | avg_rank           | avg_acc            |
|---------------------|----------------------|--------------------|--------------------|
| EnergyTS_V2.0       | 0.6122380191880193   | 1.9166666666666667 | 0.697146119179553  |
| timesfm2_xreg_early | 0.6690031495492228   | 3.566666666666667  | 0.6814488892753919 |
| chronos-bolt-base   | 0.675059380009164    | 3.6666666666666665 | 0.6811338787964563 |
| toto_151m           | 0.7045976923823263   | 4.9                | 0.6762698620557785 |
| moirai_1.1_R_large  | 0.7541134645020097   | 5.7                | 0.6708727652454117 |
| sundial_base_128m   | 0.756882361584322    | 4.366666666666666  | 0.6689460019270579 |
| tirex               | 0.7766254690582344   | 4.433333333333334  | 0.6688324908415476 |
| dummy_model         | 1.0                  | 7.45               | 0.616336212293813  |

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
        --scene wind load solar \
        --model dummy_model \
```
Use `python time_series_portal/evaluation.py --help` to view more configuration options

Generate visualization of multi-models:
```shell
python time_series_portal/visualize_multi_model_results.py \
        --dataset_path PATH/TO/YOUR/DATASET/LOCATION \
        --target_path PATH/TO/YOUR/EVALUATION_RESULT/LOCATION \
        --scene load \
        --model dummy_model toto_151m
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

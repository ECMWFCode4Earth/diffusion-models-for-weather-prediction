# WeatherDiff

![image](https://github.com/ECMWFCode4Earth/diffusion-models-for-weather-prediction/blob/main/images/z_500_lowres.gif)

## Description
This Code4Earth challenge explores the potential of Diffusion Models for weather prediction, more specificially we test it on the [WeatherBench](https://github.com/pangeo-data/WeatherBench) benchmark data set.

This repository contains functions to benchmark the diffusion models developed in [diffusion-models-for-weather-prediction](https://github.com/ECMWFCode4Earth/diffusion-models-for-weather-prediction). It builds on existing code from [WeatherBench](https://github.com/pangeo-data/WeatherBench).

## Roadmap
This repository is part of a [ECMWF Code4Earth Project](https://github.com/ECMWFCode4Earth/diffusion-models-for-weather-prediction), which takes place between May 1 2023 and September 20 2023.

## Installation

### Repository:
- The repository is formatted with black formatter and also uses pre-commit
  - make sure that pre-commit package is installed or `pip install pre-commit`
  - to set up the git hook scripts `pre-commit install`.

- The main repository has two submodules that can be installed as follows:

Clone the main repository.
Clone the `<subodules>` | Make sure you have access to them. Then:

1. `git submodule init`
2. `git submodule update`

### Data download:
- Our code requires the WeatherBench to be downloaded as described in their [repository](https://github.com/pangeo-data/WeatherBench/tree/master). We tested the 5.625° and 2.8125° resolutions.

### Setup:
- Setting up conda environments. We create 3 environments, the requirements of each of them are contained in a .yml file. Run `conda env create -f <env_config_file>` to create each environment.
  - `env_data.yml` creates an environment `WD_data` that is used to preprocess the data
  - `env_model.yml` creates an environment `WD_model` that is used to train and make prediction with machine learning models.
  - `env_eval.yml` creates an environment `WD_eval` with packages required to analyse and plot results.
- The workflow requires paths being set for a few different directories. These paths are specified in the `config/paths/` directory and make the following choices:
  - `dir_WeatherBench`: Directory the weatherBench dataset was downloaded to.
  - `dir_PreprocessedDatasets`: Preprocessed datasets get stored here
  - `dir_SavedModels`: Checkpoints and tensorboard logs are stored here
  - `dir_HydraConfigs`: When running jobs, the selected configuration files are logged here.
  - `dir_ModelOutput`: Predictions with the ML models get saved here.

### Workflow:
The workflow to train and predict with the diffusion models is as follows:
- Dataset creation: Creating a preprocessed dataset from the raw WeatherBench dataset. This can be obtained with `s1_write_dataset.py` and `submit_script_1_dataset_creation.sh` (if submitting jobscripts is required)
  - configurations for the dataset creation process and other parameter choices in the process are managed with [hydra](https://hydra.cc). The name of a configuration ("template") has to be selected when running the script, e.g. `python s1_write_data +template=<name_of_template>`. The corresponding file `<name_of_template>.yaml` should be contained in the `config/template` directory.
- Training a model: Select the appropriate script (e.g. `s2_train_pixel_diffusion`). Configuration choices are made in the `config/train.yaml` file, and experiment specific choices (model architecture, dataset, ...) are listed in the files in the `/config/experiment` directory. A experiment name has to be given, analogously the dataset creation. A model can for example be trained by `python s2_train_pixel_diffusion +experiment=<name_of_experiment>`.
    



## Contributing
Script on guidelines for contributions will be added in the future.

## Authors and acknowledgment
Participants:
- [Mohit Anand](https://github.com/melioristic)
- [Jonathan Wider](https://github.com/jonathanwider)

Mentors:
- [Jesper Dramsch](https://github.com/JesperDramsch)
- [Florian Pinault](https://github.com/floriankrb)

## License
This project is licensed under the [Apache 2.0 License](https://github.com/melioristic/benchmark/blob/main/LICENSE). The submodules contain code from external sources and are subject to the licenses included in these submodules.

## Project status
Under active developement.



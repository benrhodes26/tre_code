# Telescoping Density-Ratio Estimation
This repository contains the code used for the paper

> [Rhodes, B., Xu, K. and Gutmann, M.U., 2020. Telescoping Density-Ratio Estimation. arXiv preprint arXiv:2006.12204.](https://arxiv.org/abs/2006.12204)

This repository is no longer active. However, you are welcome to email the lead author at ben.rhodes@ed.ac.uk with questions regarding the code.

## Dependencies
The environment.yml file contains the necessary Conda/pip packages. You can easily build all dependencies via

`conda env create -f environment.yml`

## Data
Data for toy Gaussian experiments is generated automatically when running the code.

Data for the MNIST experiments can be downloaded at https://zenodo.org/record/1161203#.Wmtf_XVl8eN.

Data for the MultiOmniGlot experiments can be obtained by running `make_multiomniglot.py` (which first downloads Omniglot from https://www.tensorflow.org/datasets/catalog/omniglot).

The datasets should be stored in a directory named `density_data` (which really could have just been called `data`).

## Config files
Firstly, config files (containing all the settings for one experiment) need to be created by running the `make_configs.py` script. This script creates `.json` configs for each dataset and hyperparameter setting within a gridsearch. For instance, after running it, you can navigate to `configs/1d_gauss/model/` and you should see 6 json files, which correspond to six different settings of the hyperparameters.

You can alter the gridsearch for a particular experiment by navigating to the required function e.g. `make_mnist_configs()` and altering the final argument that gets passed into `generate_configs_for_gridsearch()`.

## Running experiments
All TRE models are trained using the `build_bridges.py` script. This script takes the command line argument `--config_path`, which should point to the `.json` config file you want to use e.g.

`python build_bridges.py --config_path=1d_gauss/model/2`

The model will be saved to `saved_models/dataset_name/time_stamp` where `time_stamp` is a time-stamp identifier for this experiment created when running `make_configs.py`.

### 1d peaked ratio experiment
In order to generate Figure 1 in the paper, we need to run

`python build_bridges.py --config_path=1d_gauss/model/0 --analyse_1d_objective=0 --analyse_single_sample_size=0`

`python build_bridges.py --config_path=1d_gauss/model/1 --analyse_1d_objective=0 --analyse_single_sample_size=0`

The required figures will be placed in `saved_models/1d_gauss/results/`.

To generate the data for Figure 2 in the paper, we need to run

`python build_bridges.py --config_path=1d_gauss/model/0 --analyse_1d_objective=0 --analyse_single_sample_size=-1`

`python build_bridges.py --config_path=1d_gauss/model/1 --analyse_1d_objective=0 --analyse_single_sample_size=-1`

This code may take a while to run (e.g. an hour, but depends on the size of the grid we use to evaluate the objective function). Note that these commands only generate the data for Figure 2 (which is saved to `saved_models/1d_gauss/time_stamp`). To actually create Figure 2, we then run the jupyter notebook in `notebooks/sample_efficiency_curves.ipynb`.

### Evaluating experiments
In order to evaluate a learned energy-based model, we run `ebm_evaluation.py` which accepts various command line arguments. In particular, we need to set `--config_path=dataset_name/time_stamp`, where `dataset_name` and `timestamp` reference a directory in `saved_models` containing a particular trained model.

In order to evaluate the representations of a model from the `SpatialMultiOmniglot`experiment, we run `representation_learning_evaluation.py`, again specifying the `--config_path`.

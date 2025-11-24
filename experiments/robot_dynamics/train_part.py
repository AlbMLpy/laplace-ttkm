import sys
import json
import argparse
import warnings
from time import time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append(str(Path.cwd().parents[1]))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from source.evaluation import rmse, nll
from source.models.LaplaceTTKM import LaplaceTTKM
from source.general_functions import create_dir_if_not_exists
from source.exp_functions import (
    train_model_time, 
    prepare_train_funct_gp,
    prepare_train_funct_vi,
    prepare_train_funct_cv,
)
from configs.robot_dynamics import ( 
    get_exp_config_vi,
    get_exp_config_cv,
    get_exp_config_gp,
)

LOG_DIR = Path('./artifacts/logs')
DATA_DIR = Path.cwd().parents[1] / 'data'
ART_DIR = Path('./artifacts/training_artifacts')

MODEL_HELP = "Choose model: 'vi', 'cv', 'gp';"

def argparse_vi():
    parser = argparse.ArgumentParser(description='Robot Dynamics Experiment')
    parser.add_argument('model', type=str, help=MODEL_HELP)
    return parser.parse_args()

def get_res_dir(args, dir_path=ART_DIR):
    return Path(dir_path / f'{args.model}/')

def get_model_spec(model: str):
    if model == 'vi':
        return LaplaceTTKM, get_exp_config_vi, prepare_train_funct_vi
    elif model == 'cv':
        return LaplaceTTKM, get_exp_config_cv, prepare_train_funct_cv
    elif model == 'gp':
        return GaussianProcessRegressor, get_exp_config_gp, prepare_train_funct_gp
    else:
        raise ValueError(f"Bad model name: {model}")
    
def save_artifacts(model_info, res_dir):
    pred_dict = dict(
        y_test=model_info.pop('y_test').tolist(),
        y_mean_test=model_info.pop('y_mean_test').tolist(),
        y_std_test=model_info.pop('y_std_test').tolist()
    )
    with open(res_dir / 'predictions.json', 'w') as f:
        json.dump(pred_dict, f)
    tg_cols = ['rmse_test', 'nll_test', 'train_time', 'prediction_time']
    res_df = pd.Series(model_info).to_frame().T[tg_cols]
    res_df.to_csv(res_dir / 'metrics.csv')

if __name__ == '__main__':
    # Prepare the dataset directory/path:
    args = argparse_vi()
    res_dir = get_res_dir(args)
    create_dir_if_not_exists(res_dir)
    data_path = DATA_DIR / 'robot_dynamics'
    # Load and preprocess the dataset:
    x_train = pd.read_csv(data_path / 'x_train.csv')
    x_test = pd.read_csv(data_path / 'x_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv')
    y_test = pd.read_csv(data_path / 'y_test.csv')
    x_train, x_test, y_train, y_test = map(jnp.array, [x_train, x_test, y_train, y_test])
    # Train the model:
    model_cls, config_f, prepare_train_funct = get_model_spec(args.model)
    train_model_f = prepare_train_funct(config_f(model_cls))
    model, model_info = train_model_time(train_model_f, x_train, y_train)
    # Test data prediction:
    start_time = time()
    y_mean_test, y_std_test = model.predict(x_test, return_std=True)
    elapsed_time = time() - start_time
    # Compute target metrics:
    model_info['prediction_time'] = elapsed_time
    model_info['rmse_test'] = rmse(y_test, y_mean_test)
    model_info['nll_test'] = nll(y_mean_test, y_std_test**2, y_test)
    model_info['y_mean_test'] = y_mean_test
    model_info['y_std_test'] = y_std_test
    model_info['y_test'] = y_test
    # Save artifacts:
    save_artifacts(model_info, res_dir)

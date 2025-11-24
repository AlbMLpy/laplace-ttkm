import gc
from time import time
from typing import Callable

import jax
import jax.numpy as jnp

import numpy as np 

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold

from .evaluation import pll, rmse
from .general_functions import full_grid

def ll_scorer(estimator, X, y):
    """ The higher the better. """
    y_pred, y_std = estimator.predict(X, return_std=True)
    return pll(y_pred, y_std**2, y)

def nrmse_scorer(estimator, X, y):
    """ The higher the better. """
    y_pred = estimator.predict(X, return_std=False)
    return -rmse(y, y_pred)

def model_factory(model_cls, params, scaler: str = 'std'):
    sc = StandardScaler if scaler == 'std' else MinMaxScaler
    return Pipeline([('scaler', sc()), ('model', model_cls(**params))])

def train_model_time(train_model: Callable, x, y):
    start_time = time()
    model, model_info = train_model(x, y)
    elapsed_time = time() - start_time
    model_info['train_time'] = elapsed_time
    return model, model_info

def prepare_train_funct_vi(config: dict, logger=None) -> Callable:
    def train_model(x, y):
        model_params = config['par_fixed']
        model = model_factory(config['model_cls'], model_params, config['scaler']) 
        model.fit(x, y)  
        if logger: logger.info(f"Training has finished:")
        return model, {}
    return train_model

def prepare_train_funct_gp(config: dict, logger=None) -> Callable:
    return prepare_train_funct_vi(config, logger)

def prepare_train_funct_cv(config: dict, logger=None) -> Callable:
    cv = RepeatedKFold(
        n_splits=config['cv_config']['n_splits'],
        n_repeats=config['cv_config']['n_repeats'], 
        random_state=config['cv_config']['seed']
    )
    options, option_names = full_grid(config['par_flexible'])
    options, n_options = list(options), len(options)
    disable = (not config['tqdm_enable'])
    def train_model(x, y):
        all_scores = []
        for idx_opt, option in tqdm(enumerate(options, 1), disable=disable, total=n_options):
            if logger: logger.info(f"##CV option {idx_opt}/{n_options} has started:")
            model_params = dict(zip(option_names, option)) | config['par_fixed']
            model = model_factory(config['model_cls'], model_params, config['scaler'])
            start_time = time()
            scores = cross_val_score(
                model, x, y, cv=cv,
                scoring=config['cv_config']['scorer'], 
                n_jobs=config['cv_config']['n_jobs'],
            )
            elapsed_time = time() - start_time
            if logger: logger.info(f"##Elapsed time={elapsed_time:.3f}")
            all_scores.append(np.mean(scores))
            # To free cache memory
            jax.clear_caches() 
            gc.collect()

        model_info = dict(
            scores=all_scores, 
            **{k:list(v) for k, v in zip(option_names, list(zip(*options)))}
        )
        best_option = options[np.argmax(all_scores)]
        model_params = dict(zip(option_names, best_option)) | config['par_fixed']
        model = model_factory(config['model_cls'], model_params, config['scaler'])
        model.fit(x, y)  
        return model, model_info
    return train_model

def get_stats_several_trials(
    load_data: Callable,
    train_model: Callable,
    tracker: object,
    n_trials: int = 10,
    tqdm_disable: bool = False,
    logger = None,
) -> None:
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable):
        if logger: logger.info(f"#Trial {trial}/{n_trials} has started.")
        x_train, x_test, y_train, y_test = load_data(split_seed=trial)
        x_train, x_test, y_train, y_test = map(jnp.array, [x_train, x_test, y_train, y_test])
        model, model_info = train_model_time(train_model, x_train, y_train)
        tracker.track(trial, x_train, x_test, y_train, y_test, model, model_info)

from typing import Callable

MODELS = ['vi', 'cv']
FMAP, FSHIFT = 'poly_norm', 0
DATASETS = [
    'yacht', 'energy', 'boston', 'concrete', 'kin8nm', 'naval', 'protein'
]
N_TRIALS = 10
DATA_SEED = 15
TEST_SIZE = 0.1
RES_NAME, CV_NAME = 'results', 'cv_train'
TRANSFORM_X, TRANSFORM_Y, SCALER = False, True, 'std'

PD_MODE = 'la'
N_EPOCH_VI = 5
D_ALS_CORE = 0
HESS_THS = 1e-3
MODEL_SEED = 13
PD_SAMPLES = 30
BETA_E_SAMPLES = 10
BETA_E_LIST = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
GAMMA_W_LIST = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
CV_CONFIG = dict(n_splits=3, n_repeats=2, seed=1, n_jobs=1)

data2params = dict(
    yacht=dict(m_order=8, tt_ranks=(1, *(2,)*5, 1), n_epoch=20), 
    energy=dict(m_order=16, tt_ranks=(1, *(2,)*7, 1), n_epoch=20),
    boston=dict(m_order=8, tt_ranks=(1, *(2,)*12, 1), n_epoch=20),
    concrete=dict(m_order=12, tt_ranks=(1, *(3,)*7, 1), n_epoch=20),
    kin8nm=dict(m_order=8, tt_ranks=(1, *(3,)*7, 1), n_epoch=5),
    naval=dict(m_order=8, tt_ranks=(1, *(3,)*15, 1), n_epoch=5),
    protein=dict(m_order=8, tt_ranks=(1, *(3,)*8, 1), n_epoch=5),
)

def get_exp_config_vi(
    data_name: str, 
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str,  
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    data_params = data2params[data_name]
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            tt_ranks=data_params['tt_ranks'], fmap=fmap, 
            m_order=data_params['m_order'], n_epoch=data_params['n_epoch'], 
            beta_e=None, gamma_w=None, pd_mode=PD_MODE, hess_type=D_ALS_CORE, 
            hess_th=HESS_THS, seed=MODEL_SEED, n_epoch_vi=N_EPOCH_VI,
            pd_samples=PD_SAMPLES, beta_e_samples=BETA_E_SAMPLES, 
            d_core_als=D_ALS_CORE,
        ),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples, d_dim=d_dim, 
            test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_cv(
    data_name: str,
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    data_params = data2params[data_name]
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            tt_ranks=data_params['tt_ranks'], fmap=fmap, 
            m_order=data_params['m_order'], n_epoch=data_params['n_epoch'], 
            pd_mode=PD_MODE, hess_type=D_ALS_CORE, 
            hess_th=HESS_THS, seed=MODEL_SEED, n_epoch_vi=N_EPOCH_VI,
            pd_samples=PD_SAMPLES, beta_e_samples=BETA_E_SAMPLES, 
            d_core_als=D_ALS_CORE,
        ),
        par_flexible=dict(beta_e=BETA_E_LIST, gamma_w=GAMMA_W_LIST),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples, d_dim=d_dim, 
            test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

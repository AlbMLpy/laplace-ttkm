import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[0]))

import sklearn.gaussian_process.kernels as gp_kern

from source.features import FFeature
from source.exp_functions import ll_scorer

M_ORDER = 8
N_EPOCH = 3
PD_MODE = 'la'
SCALER = 'std'
D_CORE_ALS = 0
N_EPOCH_VI = 2
HESS_THS = 1e-7
MODEL_SEED = 13
PD_SAMPLES = 30
BETA_E_SAMPLES = 10
FMAP = FFeature(p_scale=105.0)
BETA_E_LIST = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
GAMMA_W_LIST = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
TT_RANKS = [1] + [3 for _ in range(17)] + [1]

def get_exp_config_vi(model_cls) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            tt_ranks=TT_RANKS, fmap=FMAP, m_order=M_ORDER, 
            n_epoch=N_EPOCH, beta_e=None, gamma_w=None, pd_mode=PD_MODE, 
            hess_type=D_CORE_ALS, hess_th=HESS_THS, seed=MODEL_SEED, 
            n_epoch_vi=N_EPOCH_VI, pd_samples=PD_SAMPLES, 
            beta_e_samples=BETA_E_SAMPLES, d_core_als=D_CORE_ALS,
        ),
    )

def get_exp_config_cv(model_cls) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            tt_ranks=TT_RANKS, fmap=FMAP, m_order=M_ORDER, 
            n_epoch=N_EPOCH, pd_mode=PD_MODE, 
            hess_type=D_CORE_ALS, hess_th=HESS_THS, seed=MODEL_SEED, 
            n_epoch_vi=N_EPOCH_VI, pd_samples=PD_SAMPLES, 
            beta_e_samples=BETA_E_SAMPLES, d_core_als=D_CORE_ALS,
        ),
        par_flexible=dict(beta_e=BETA_E_LIST, gamma_w=GAMMA_W_LIST),
        cv_config=dict(n_splits=3, n_repeats=2, seed=1, n_jobs=1, scorer=ll_scorer),
        tqdm_enable=True,
    )

def get_exp_config_gp(model_cls) -> dict:
    l_scale, sigma_f, sigma_n = 0.4706, 2.8853, 0.6200 # Init params 
    k1 = gp_kern.ConstantKernel(constant_value=sigma_f**2)
    k2 = gp_kern.RBF(length_scale=l_scale)
    k3 = gp_kern.WhiteKernel(sigma_n**2)
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            kernel=k1*k2 + k3, 
            optimizer='fmin_l_bfgs_b', 
            random_state=0
        ),
    )

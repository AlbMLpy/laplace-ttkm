import sys
import pytest

import jax
import jax.numpy as jnp
from jax import jit, grad
jax.config.update("jax_enable_x64", True)

import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.model_selection import train_test_split

sys.path.append('./')

from source.features import PPNFeature
from source.trackers import GradTracker
from source.models.LaplaceTTKM import LaplaceTTKM
from source.data_functions import generate_x3_data, scale_data
from source.evaluation import l2_loss_tt, l2_loss_vec, nll, rmse

DATA_PARAMS = [(500, 6, 13), (1000, 3, 14), (500, 16, 14)]

@pytest.fixture(params=DATA_PARAMS, scope='module')
def synthetic_train_data(request):
    n_samples, d_dim, seed = request.param
    x_train, y_train, _ = generate_x3_data(
        n_samples, d_dim, std_err=3, seed=seed)
    return x_train, y_train

@pytest.fixture(scope='module')
def trained_model(synthetic_train_data):
    """ 
    In this setting gamma_w and beta_w are known (n_epoch_vi=1).
    """
    x_train, y_train = synthetic_train_data
    beta_e, gamma_w = 1e-1, 1e-3
    grad_w = jit(grad(l2_loss_vec, argnums=0), static_argnums=(4, 7, 8))
    tracker = GradTracker(
        x_train, y_train, beta_e, gamma_w, l2_loss_tt, grad_w)

    d_dim, rank = x_train.shape[1], 2
    tt_ranks = tuple([1] + [rank for _ in range(d_dim-1)] + [1])
    m_order, n_epoch = 4, 100
    hess_d_core, hess_th = 1, None

    model = LaplaceTTKM(
        tt_ranks=tt_ranks, fmap=PPNFeature(), m_order=m_order, n_epoch=n_epoch,
        beta_e=beta_e, gamma_w=gamma_w, pd_mode='la', hess_type=hess_d_core,
        hess_th=hess_th, seed=13, n_epoch_vi=1, pd_samples=30,
        beta_e_samples=10, tracker=tracker,
    )
    model.fit(x_train, y_train)
    return model, tracker.res_dict

class TestLaplaceTTKMTraining:
    def test_loss_decrease(self, trained_model):
        _, res = trained_model
        loss_vals = jnp.array(res['loss'])
        assert jnp.all(loss_vals == jnp.sort(loss_vals)[::-1]), "Loss should decrease over training."

    def test_grad_norm_decrease(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        assert gn_vals[0] > gn_vals[-1], "Gradient norm should decrease over training."

    def test_grad_moving_avg_decrease(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        mov_avg = np.convolve(gn_vals, np.ones(10)/10, mode='valid')
        assert mov_avg[0] > mov_avg[-1], "Moving average should decrease."


@pytest.fixture(scope='module')
def synthetic_train_test_data():
    X, y = make_friedman2(n_samples=600, noise=1, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.16, random_state=42)
    return scale_data(
        x_train, x_test, y_train, y_test, True, False, 'std'
    )

@pytest.fixture(scope='module')
def trained_model_vi(synthetic_train_test_data):
    """ 
    In this setting beta_w is known (n_epoch_vi=1).
    """
    x_train, x_test, y_train, y_test = synthetic_train_test_data
    beta_e, gamma_w = None, None
    grad_w = jit(grad(l2_loss_vec, argnums=0), static_argnums=(4, 7, 8))
    tracker = GradTracker(
        x_train, y_train, beta_e, gamma_w, l2_loss_tt, grad_w)
    
    d_dim, rank = x_train.shape[1], 4
    tt_ranks = tuple([1] + [rank for _ in range(d_dim-1)] + [1])
    hess_d_core, hess_th = 1, None
    model = LaplaceTTKM(
        tt_ranks=tt_ranks, fmap=PPNFeature(), m_order=40, n_epoch=5,
        beta_e=beta_e, gamma_w=gamma_w, pd_mode='la', 
        hess_type=hess_d_core, hess_th=hess_th, 
        seed=13, n_epoch_vi=5, pd_samples=30, 
        beta_e_samples=10, tracker=tracker,
    )
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test, tracker.res_dict

class TestLaplaceTTKMRealData:
    def test_predictions_train(self, trained_model_vi):
        model, x_train, _, y_train, _, res = trained_model_vi
        ys_train, ys_std_train = model.predict(x_train, True)
        loss_vals_ma = np.convolve(res['loss'], np.ones(10)/10, mode='valid')
        assert jnp.all(loss_vals_ma == jnp.sort(loss_vals_ma)[::-1]), "Loss should decrease over training."
        assert nll(ys_train, ys_std_train**2, y_train) < 15, "Check train NLL."
        assert rmse(ys_train, y_train) < 60, "Check train RMSE."

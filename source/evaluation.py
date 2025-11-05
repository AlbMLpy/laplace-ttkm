import jax
import jax.numpy as jnp

from sklearn.metrics import root_mean_squared_error

from .matrix_operations import vec2tt
from .prob_functions import w_sample_diag
from .model_functionality_tt import predict_score_tt

def rmse(y1, y2): 
    return root_mean_squared_error(y1, y2)

def pll(y_pred, var_pred, y_true):
    """ Computes the predictive log-likelihood. """
    log_likelihoods = -0.5 * jnp.log(2 * jnp.pi * var_pred) - ((y_true - y_pred) ** 2) / (2 * var_pred)
    return jnp.mean(log_likelihoods).item()

def nll(y_pred, var_pred, y_true):
    """ Computes the negative log-likelihood. """
    return -pll(y_pred, var_pred, y_true)

def norm_frob(x):
    return jnp.sqrt((x * x).sum())

def l2_loss_tt(w_tt, kd, x, y, fmap, gamma_w, beta_e):
    scores = predict_score_tt(x, kd, w_tt, fmap)        
    return (
        (0.5*(beta_e*jnp.sum((y - scores)**2) 
        + gamma_w*sum([(wk * wk).sum() for wk in w_tt])))
    )

def l2_loss_vec(w_vec, kd, x, y, fmap, gamma_w, beta_e, m_order, rank_list):
    w_tt = vec2tt(w_vec, m_order, rank_list)
    return l2_loss_tt(w_tt, kd, x, y, fmap, gamma_w, beta_e)

def l2_loss_kl(w_mean, w_std, m_order, rank_list, kd, x, y, fmap, gamma_w, beta_e, key, n_samples):
    """ L2 loss with closed form KL divergence. """
    w_var = w_std**2
    kl_term = gamma_w * (w_var.sum() + (w_mean**2).sum()) - jnp.sum(jnp.log(w_var))
    keys = jax.random.split(key, n_samples)

    def sample_loss(subkey):
        w_sample = w_sample_diag(w_mean, w_std, subkey)
        scores = predict_score_tt(x, kd, vec2tt(w_sample, m_order, rank_list), fmap) 
        return beta_e * jnp.sum((y - scores)**2)
    
    likelihood_term = jax.vmap(sample_loss)(keys).sum()
    return 0.5 * (likelihood_term + kl_term)

from typing import Optional, Union

import numpy as np 
from jax import Array
import jax.numpy as jnp

from .features import FeatureMap
from .matrix_operations import tt2vec, vec2tt
from .model_functionality_tt import (
    get_tt_size,
    get_tt_ranks,
    get_tt_sqnorm,
    get_tt_ind_shift,
    predict_score_tt, 
    predict_score_tt_linear,
)

EPS_DIV = 1e-8

def init_beta_e(beta_e: Optional[float] = None):
    beta_e_upd, cn, dn = beta_e is None, None, None
    if beta_e_upd:
        cn, dn = 1, 1
        beta_e = cn / dn
    return beta_e, beta_e_upd, cn, dn

def init_gamma_w(gamma_w: Optional[float] = None):
    gamma_w_upd, an, bn = gamma_w is None, None, None
    if gamma_w_upd:
        an, bn = 1, 1
        gamma_w = an / bn
    return gamma_w, gamma_w_upd, an, bn

def update_gamma_w(a, b, w_cholesky, w):
    w_cov_diag = (w_cholesky * w_cholesky).sum(axis=1)
    a += 0.5 * get_tt_size(w)
    b += 0.5 * (get_tt_sqnorm(w) + w_cov_diag.sum())
    return a, b, a / max(b, EPS_DIV)

def w_sample(w_mean_vec, w_cholesky, ind, shift, seed: Optional[int] = None):
    """
    Sampling from a Gaussian Posterior.
    """
    z = jnp.array(np.random.RandomState(seed).randn(w_cholesky.shape[1]))
    w_sample = w_mean_vec.copy()
    w_sample = w_sample.at[ind:ind + shift].set(
        w_sample[ind:ind + shift] + w_cholesky.dot(z))
    return w_sample

def get_scores(
    w_mean_vec: Array, 
    w_vec_sample: Array, 
    m_order: int,
    tt_ranks: tuple[int],
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    pd_mode: str,
) -> Array:
    if pd_mode == 'lla':
        scores = predict_score_tt_linear(
            x, kd, vec2tt(w_vec_sample, m_order, tt_ranks), fmap, 
            vec2tt(w_mean_vec, m_order, tt_ranks),
        )
    elif pd_mode == 'la':
        scores = predict_score_tt(
            x, kd, vec2tt(w_vec_sample, m_order, tt_ranks), fmap)
    return scores

def _preprocess(w_tt: list[Array], hess_type: Union[str, int]):
    w_mean_vec = tt2vec(w_tt)
    m_order, tt_ranks = w_tt[0].shape[1], get_tt_ranks(w_tt)
    if isinstance(hess_type, int):
        ind, shift = get_tt_ind_shift(hess_type, m_order, tt_ranks)
    else:
        ind, shift = 0, get_tt_size(w_tt)
    return w_mean_vec, m_order, tt_ranks, ind, shift

def _sample_scores(
    idx_s, 
    w_mean_vec, 
    w_cholesky, 
    ind, 
    shift, 
    m_order, 
    tt_ranks, 
    kd, 
    x, 
    fmap, 
    pd_mode, 
    seed
):
    sample_seed = None if seed is None else seed + idx_s
    w_vec_sample = w_sample(
        w_mean_vec, w_cholesky, ind, shift, sample_seed)
    return get_scores(
        w_mean_vec, w_vec_sample, m_order, tt_ranks, kd, x, fmap, pd_mode)

def predict_std(
    w_tt: list[Array], 
    w_cholesky: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    beta_e: Optional[float],
    hess_type: Union[str, int], 
    pd_mode: str, 
    n_samples: int,
    seed: Optional[int],
) -> Array:
    w_mean_vec, m_order, tt_ranks, ind, shift = _preprocess(w_tt, hess_type)
    preds = []
    for idx_s in range(n_samples):
        scores = _sample_scores(
            idx_s, w_mean_vec, w_cholesky, ind, shift, 
            m_order, tt_ranks, kd, x, fmap, pd_mode, seed
        )
        preds.append(scores[:, None])
    pred_std = jnp.std(jnp.hstack(preds), axis=1)
    if beta_e is None:
        return pred_std
    else:
        return pred_std + 1 / jnp.sqrt(beta_e)
    
def update_beta_e(
    c: float, 
    d: float, 
    w_tt: list[Array], 
    w_cholesky: Array, 
    kd: int, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    hess_type: Union[str, int],
    pd_mode: str, 
    n_samples: int,
    seed: Optional[int],
):
    w_mean_vec, m_order, tt_ranks, ind, shift = _preprocess(w_tt, hess_type)
    mean_train_err = 0.0
    for idx_s in range(n_samples):
        scores = _sample_scores(
            idx_s, w_mean_vec, w_cholesky, ind, shift, 
            m_order, tt_ranks, kd, x, fmap, pd_mode, seed
        )
        mean_train_err += jnp.sum((y - scores)**2)
    mean_train_err /= n_samples
    c += 0.5 * x.shape[0]
    d += 0.5 * mean_train_err
    return c, d, c / max(d, EPS_DIV)

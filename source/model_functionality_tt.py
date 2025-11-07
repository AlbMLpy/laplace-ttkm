from functools import partial
from typing import Optional, Union

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array, jit

from .features import FeatureMap
from .matrix_operations import khatri_rao_row

def init_weights_tt(
    m_order: int, 
    rank_list: list[int], 
    q_base: Optional[int] = None, 
    init_type: Optional[str] = None, 
    seed: Optional[int] = None,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[list[Array], int]:
    """
    Initialize TT model weights using Gaussian Distribution 
    and optional normalization strategies. Use 'q_base' parameter 
    to generate quantized representation of weights.
    
    Note: rank_list[0] and rank_list[-1] ranks are supposed to be ones.
    """
    if (m_order & (m_order - 1)) and q_base:
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}. ")
    if rank_list[0] != 1 or rank_list[-1] != 1:
        raise ValueError(f"rank_list[0] and rank_list[-1] ranks are supposed to be ones. ")
    key = jax.random.PRNGKey(seed or np.random.randint(1e10))
    keys = jax.random.split(key, len(rank_list) - 1)
    if q_base:
        raise NotImplementedError()
    else:
        kd = 1
        weights = [
            jax.random.normal(k, (r1, m_order, r2), dtype=dtype)
            for k, r1, r2 in zip(keys, rank_list[:-1], rank_list[1:])
        ]
    if init_type == 'k_mtx': 
        raise NotImplementedError()
    elif init_type == 'kj_vec': 
        raise NotImplementedError()
    return weights, kd

@partial(jit, static_argnums=(3,))
def prepare_buffer_tt(
    x: Array, 
    kd: int, 
    w_tt: list[Array],
    fmap: FeatureMap,
) -> list[Array]:
    """ 
    Prepare a buffer with intermediate calculations.
    
    Note: For computationally efficient ALS update.
    """
    buf_size = len(w_tt) - 1 # buffer has D-1 elements
    buf = [d for d in range(buf_size)]
    for ind in range(buf_size, 0, -1): 
        wk, (k, q) = w_tt[ind], divmod(ind, kd) # q starts from zero -> for fmap
        if ind == buf_size:
            buf[ind-1] = jnp.einsum(
                'npi,rip->nr', fmap(x[:, k], q)[:, None, :], wk)
        else:
            g_hat = jnp.einsum('ni,nr->nri', fmap(x[:, k], q), buf[ind])
            buf[ind-1] = jnp.einsum('npi,rip->nr', g_hat, wk)
    return buf

@partial(jit, static_argnums=(3,))
def predict_score_tt(
    x: Array, 
    kd: int, 
    w_tt: list[Array], 
    fmap: FeatureMap,
) -> Array:
    """ 
    Generate prediction scores for TTKM model. 
    """
    for ind in range(len(w_tt)-1):
        wk, (k, q) = w_tt[ind], divmod(ind, kd) # q starts from zero -> for fmap
        if ind == 0:
            g_hat = fmap(x[:, k], q)[:, None, :]
        g_core = jnp.einsum('nri,rip->np', g_hat, wk)
        g_hat = jnp.einsum('ni,nr->nri', fmap(x[:, k+1], q), g_core) # k+1? ind??
    pred = jnp.einsum('nri,rip->np', g_hat, w_tt[-1]).squeeze()
    return jnp.real(pred)

@partial(jit, static_argnums=(3,))
def predict_score_tt_linear(
    x: Array, 
    kd: int, 
    w_tt: list[Array], 
    fmap: FeatureMap,
    w_tt_opt: list[Array],
) -> Array:
    """ 
    Generate prediction scores for linearized TTKM model. 
    """
    raise NotImplementedError()

#@jit
def _prepare_system_tt(
    left_mtx: Optional[Array],
    fk_mtx: Array,
    right_mtx: Optional[Array], 
    y: Array,
) -> tuple[Array, Array]:
    if left_mtx is None:
        Ak = khatri_rao_row(fk_mtx, right_mtx)
    elif right_mtx is None:
        Ak = khatri_rao_row(left_mtx, fk_mtx)
    else:
        Ak = khatri_rao_row(khatri_rao_row(left_mtx, fk_mtx), right_mtx)
    return Ak.T.conj().dot(Ak), Ak.T.conj().dot(y)

#@jit
def prepare_system_tt(
    ind: int,
    buf: list[Array],
    fk_mtx: Array,
    y: Array,
) -> tuple[Array, Array]:
    """
    Prepare local linear system of equations for one TT core.
    """
    if ind == 0:
        A, b = _prepare_system_tt(buf[ind], fk_mtx, None, y)
    elif ind == len(buf):
        A, b = _prepare_system_tt(None, fk_mtx, buf[ind-1], y)
    else:
        A, b = _prepare_system_tt(buf[ind], fk_mtx, buf[ind-1], y)
    return A, b

def postprocess_tt(bottom_up, ind, buf, fk_mtx, wk):
    """
    Update the corresponding buffer matrices after the ALS update.
    """
    if bottom_up: # d=0, ..., D-1
        if ind == 0:
            buf[ind] = jnp.einsum(
                'nri,rip->np', fk_mtx[:, None, :], wk)
        else:
            g_hat = jnp.einsum('ni,nr->nri', fk_mtx, buf[ind-1])
            buf[ind] = jnp.einsum('nri,rip->np', g_hat, wk)
    else: # d=D-1, ..., 1
        if ind == len(buf):
            buf[ind-1] = jnp.einsum(
                'npi,rip->nr', fk_mtx[:, None, :], wk)
        else:
            g_hat = jnp.einsum('ni,nr->nri', fk_mtx, buf[ind])
            buf[ind-1] = jnp.einsum('npi,rip->nr', g_hat, wk)

#@partial(jit, static_argnums=(4,))
def update_weights_tt(
    w_tt: list[Array], 
    kd: int, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float, 
    buf: list[Array],
    tracker: Optional[object] = None,
) -> list[Array]:
    """ 
    Full update of all the model's TT cores (one sweep/epoch).

    Note: TT cores are updated 2 times except for the first and the last ones.
    """
    d_dim, bottom_up = len(w_tt), True
    upd_sequence = list(range(d_dim)) + list(reversed(range(1, d_dim-1)))
    for ind in upd_sequence:
        if ind == d_dim - 1: bottom_up = False
        # Prepare matrices for a linear system:
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        wk, fk_mtx = w_tt[ind], fmap(x[:, k], q)
        A, b = prepare_system_tt(ind, buf, fk_mtx, y)
        # Solve a linear system:
        A += gamma_w / beta_e * jnp.eye(A.shape[0])
        sol = jnp.linalg.solve(A, b)
        w_tt[ind] = wk = jnp.array(sol.reshape(wk.shape, order='F')) 
        # Postprocess the buffer:
        postprocess_tt(bottom_up, ind, buf, fk_mtx, wk)
        if tracker: tracker.track(w_tt, kd, fmap)
    return w_tt

def als_tt(
    w_tt: list[Array], 
    kd: int,  
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    n_epoch: int, 
    gamma_w: float, 
    beta_e: float, 
    tracker: Optional[object] = None,
) -> list[Array]:
    """
    Compute "optimal" TT-based model weights using ALS optimization algorithm.
    """
    buf = prepare_buffer_tt(x, kd, w_tt, fmap)
    if tracker: tracker.track(w_tt, kd, fmap, beta_e, gamma_w)
    for _ in range(n_epoch):
        w_tt = update_weights_tt(w_tt, kd, x, y, fmap, gamma_w, beta_e, buf)
        if tracker: tracker.track(w_tt, kd, fmap, beta_e, gamma_w)
    return w_tt

@partial(jit, static_argnums=(3,))
def jacob_tt(
    w_tt: list[Array], 
    kd: int, 
    x: Array, 
    fmap: FeatureMap,
) -> Array:
    """
    Compute the Jacobian matrix of the TTKM model prediction function.
    """
    raise NotImplementedError()

def hess_cov_estimation(
    w_tt: list[Array], 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float, 
    hess_type: Union[str, int], 
    hess_th: float = 1e-3,
) -> tuple[Array, Optional[Array], Optional[Array]]:
    """
    Compute the Hessian approximation of the L2 loss in the TTKM model setting.
    """
    if hess_type == 'gauss_newton':
        raise NotImplementedError()
    elif hess_type == 'block':
        raise NotImplementedError()
    elif hess_type == 'mf':
        raise NotImplementedError()
    elif isinstance(hess_type, int):
        w_hess = hess_one_core_tt(
            hess_type, w_tt, kd, x, fmap, gamma_w, beta_e)
    else:
        raise ValueError(f'Bad hess_type: {hess_type}')
    
    if hess_th:
        cov_f = partial(low_rank_cov_estimation, threshold=hess_th)
    else:
        cov_f = cov_estimation
    
    try: 
        w_cov, w_cholesky = cov_f(w_hess)
        success = True
    except Exception as e: 
        success = False

    if success: 
        return w_hess, w_cov, w_cholesky
    else:
        return w_hess, None, None
    
def prepare_p_core_left(
    d_core: int, 
    x: Array, 
    kd: int, 
    w_tt: list[Array],
    fmap: FeatureMap,
) -> Optional[list[Array]]:
    if d_core == 0: return None
    if d_core < 0 or d_core > len(w_tt): raise ValueError()
    
    for ind in range(d_core):
        wk, (k, q) = w_tt[ind], divmod(ind, kd) # q starts from zero -> for fmap
        if ind == 0:
            g_hat = fmap(x[:, k], q)[:, None, :]
        g_core = jnp.einsum('nri,rip->np', g_hat, wk)
        g_hat = jnp.einsum('ni,nr->nri', fmap(x[:, k+1], q), g_core)
    return g_core

def prepare_q_core_right(
    d_core: int, 
    x: Array, 
    kd: int, 
    w_tt: list[Array],
    fmap: FeatureMap,
) -> Optional[list[Array]]:
    if d_core == len(w_tt)-1: return None
    if d_core < 0 or d_core >= len(w_tt): raise ValueError()

    for ind in range(len(w_tt)-1, d_core, -1):
        wk, (k, q) = w_tt[ind], divmod(ind, kd) # q starts from zero -> for fmap
        if ind == len(w_tt)-1:
            g_hat = fmap(x[:, k], q)[:, None, :]
        g_core = jnp.einsum('npi,rip->nr', g_hat, wk)
        g_hat = jnp.einsum('ni,nr->nri', fmap(x[:, k-1], q), g_core)
    return g_core
    
#@partial(jit, static_argnums=(3,))
def hess_one_core_tt(
    d_core: int, 
    w_tt: list[Array], 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float,
) -> Array:
    """
    Estimation of the Hessian related to one TT-core.
    """
    k, q = divmod(d_core, kd)
    fk_mtx = fmap(x[:, k], q)
    p_mtx = prepare_p_core_left(d_core, x, kd, w_tt, fmap)
    q_mtx = prepare_q_core_right(d_core, x, kd, w_tt, fmap)
    a_mtx, _ = _prepare_system_tt(
        q_mtx, fk_mtx, p_mtx, jnp.zeros(x.shape[0])
    )
    return beta_e*a_mtx + gamma_w*jnp.eye(a_mtx.shape[0])

def cov_estimation(w_hess: Array) -> tuple[Array, Array]:
    """
    Compute corresponding covariance matrix and 
    Cholesky factor based on a provided Hessian matrix.
    """
    w_cholesky = np.linalg.cholesky(w_hess)
    w_cholesky = np.linalg.pinv(w_cholesky.T)
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def low_rank_cov_estimation(w_hess: Array, threshold: float = 1e-3) -> Array:
    """
    Compute corresponding low-rank covariance matrix and 
    Cholesky factor based on a provided Hessian matrix.
    """
    s, u = np.linalg.eigh(w_hess)
    mask = s >= threshold
    w_cholesky = u[:, mask] * (1/jnp.sqrt(s[mask]))[None, :]
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def get_tt_size(w_tt: list[Array]) -> int:
    return sum([wd.size for wd in w_tt])

def get_tt_sqnorm(w_tt: list[Array]) -> int:
    return sum([(wd * wd).sum() for wd in w_tt])

def get_tt_ranks(w_tt: list[Array]) -> tuple[int]:
    return tuple([wd.shape[0] for wd in w_tt] + [1])

def get_tt_ind_shift(d_core: int, m_order: int, tt_ranks: tuple[int]):
    ind = sum([tt_ranks[i]*m_order*tt_ranks[i+1] for i in range(d_core)])
    shift = tt_ranks[d_core]*m_order*tt_ranks[d_core+1]
    return ind, shift

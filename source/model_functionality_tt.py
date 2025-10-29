from typing import Optional
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from .features import FeatureMap

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
        g_hat = jnp.einsum('ni,nr->nri', fmap(x[:, k+1], q), g_core) 
    pred = jnp.einsum('nri,rip->np', g_hat, w_tt[-1]).squeeze()
    return jnp.real(pred)

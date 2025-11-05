from functools import partial

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

@jit
def khatri_rao_row(a: ArrayLike, b: ArrayLike) -> Array:
    return vmap(jnp.kron)(a, b)

@jit
def tt2vec(w_tt: list[Array]) -> Array:
    return jnp.concatenate([wk.reshape(-1, order='F') for wk in w_tt])

@partial(jit, static_argnums=[1, 2])
def vec2tt(w_vec: Array, m_order: int, rank_list: tuple[int]) -> list[Array]:
    w_tt, ind = [], 0
    for i in range(len(rank_list)-1):
        r1, r2 = rank_list[i], rank_list[i+1]
        shift = r1 * m_order * r2
        w_tt.append(w_vec[ind:ind + shift].reshape(r1, m_order, r2, order='F'))
        ind += shift
    return w_tt

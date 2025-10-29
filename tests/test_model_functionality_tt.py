import sys
import unittest
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, jacrev, hessian
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.matrix_operations import khatri_rao_row
from source.features import PPFeature, prepare_fmap
from source.model_functionality_tt import (
    init_weights_tt,
    predict_score_tt,
    prepare_buffer_tt,
)

def prepare_a_mtx(ind: int, buf: list, fk_mtx):
    if ind == 0: 
        return khatri_rao_row(buf[ind], fk_mtx)
    elif ind == len(buf): 
        return khatri_rao_row(fk_mtx, buf[ind-1])
    else: 
        return khatri_rao_row(khatri_rao_row(buf[ind], fk_mtx), buf[ind-1])

class TestModelFunctionality(unittest.TestCase):
    def test_init_weights_tt(self):
        m_order = 5
        rank_list = [1, 2, 3, 1] # d_dim = 3
        seed = 12
        tt_weights, _ = init_weights_tt(m_order, rank_list, seed=seed)
        expected = [(1, 5, 2), (2, 5, 3), (3, 5, 1)]
        actual = [tt.shape for tt in tt_weights]
        self.assertTrue(actual == expected)

    def test_predict_score_tt(self):
        m_order = 2
        fmap, _ = prepare_fmap(PPFeature(), m_order, False)
        x = jnp.array([[1, 2, 3], [2, 1, 4], [1, 1, 1]])
        rank_list = [1, 3, 3, 1]
        seed = 12
        tt_weights, kd = init_weights_tt(m_order, rank_list, seed=seed)

        phi = khatri_rao_row(fmap(x[:, 1], None), fmap(x[:, 0], None))
        phi = khatri_rao_row(fmap(x[:, 2], None), phi)
        tt_full = np.empty((m_order,)*3)
        for k in range(m_order):
            for j in range(m_order):
                for i in range(m_order):
                    tt_full[i, j, k] = (
                        tt_weights[0][:, i, :].dot(tt_weights[1][:, j, :])
                    ).dot(tt_weights[2][:, k, :]).squeeze()
        tt_full = jnp.array(tt_full)

        expected = phi.dot(tt_full.reshape(-1, order='F'))
        actual = predict_score_tt(x, kd, tt_weights, fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_prepare_buffer_tt(self):
        m_order = 2
        fmap, _ = prepare_fmap(PPFeature(), m_order, False)
        x = jnp.array([[1, 2, 3], [2, 1, 4], [1, 1, 1]])
        rank_list = [1, 4, 5, 1]
        seed = 12
        tt_weights, kd = init_weights_tt(m_order, rank_list, seed=seed)
        buf = prepare_buffer_tt(x, kd, tt_weights, fmap)
        # Test shapes:
        expected = [(x.shape[0], r) for r in rank_list[1:-1]]
        actual = [b.shape for b in buf]
        self.assertTrue(np.allclose(actual, expected))
        # Test prediction ability:
        d = 0
        a_mtx = prepare_a_mtx(d, buf, fmap(x[:, d], None))
        wd_vec = tt_weights[d].reshape(-1, order='F')
        expected = predict_score_tt(x, kd, tt_weights, fmap)
        actual = a_mtx.dot(wd_vec)
        self.assertTrue(np.allclose(actual, expected))


import sys
import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, jacrev, hessian
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.trackers import LossTracker
from source.features import PPFeature, prepare_fmap
from source.evaluation import l2_loss_tt, l2_loss_vec
from source.matrix_operations import khatri_rao_row, tt2vec
from source.model_functionality_tt import (
    als_tt,
    init_weights_tt,
    predict_score_tt,
    hess_one_core_tt,
    prepare_buffer_tt,
    update_weights_tt,
    prepare_p_core_left,
    prepare_q_core_right,
)

def prepare_a_mtx(ind: int, buf: list, fk_mtx):
    if ind == 0: 
        return khatri_rao_row(buf[ind], fk_mtx)
    elif ind == len(buf): 
        return khatri_rao_row(fk_mtx, buf[ind-1])
    else: 
        return khatri_rao_row(khatri_rao_row(buf[ind], fk_mtx), buf[ind-1])

def prepare_a_mtx_v2(left_mtx, fk_mtx, right_mtx):
    if left_mtx is None:
        return khatri_rao_row(fk_mtx, right_mtx)
    elif right_mtx is None:
        return khatri_rao_row(left_mtx, fk_mtx)
    else:
        return khatri_rao_row(khatri_rao_row(left_mtx, fk_mtx), right_mtx)
    
def hess_full_jax(w_tt, kd, x, y, fmap, gamma_w, beta_e, m_order, rank_list):
    hess_f = jit(hessian(l2_loss_vec, argnums=0), static_argnums=(4, 7, 8))
    w_vec = tt2vec(w_tt)
    hw = hess_f(w_vec, kd, x, y, fmap, gamma_w, beta_e, m_order, rank_list)
    return hw

class TestModelFunctionality(unittest.TestCase):
    def setUp(self):
        self.x = jnp.array(
            [
                    [1, 2, 3], 
                    [2, 1, 4], 
                    [1, 1, 1],
            ]
        )
        self.y = jnp.array([1, 2, 3])
        self.m_order = 4
        self.rank_list = (1, 4, 5, 1)
        self.seed = 12
        self.gamma_w = 0.1
        self.beta_e = 0.1 
        self.n_epoch = 10
        self.w_tt, self.kd = init_weights_tt(
                self.m_order, self.rank_list, seed=self.seed
        )
        self.fmap, _ = prepare_fmap(PPFeature(), self.m_order, False)

    def test_init_weights_tt(self):
        tt_weights, _ = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)

        expected = [
            (self.rank_list[i], self.m_order, self.rank_list[i+1]) 
            for i in range(len(self.rank_list)-1)
        ]
        actual = [tt.shape for tt in tt_weights]
        self.assertTrue(actual == expected)

    def test_predict_score_tt(self):
        tt_weights, kd = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)
        phi = khatri_rao_row(
            self.fmap(self.x[:, 1], None), self.fmap(self.x[:, 0], None))
        phi = khatri_rao_row(self.fmap(self.x[:, 2], None), phi)
        tt_full = np.empty((self.m_order,)*3)
        for k in range(self.m_order):
            for j in range(self.m_order):
                for i in range(self.m_order):
                    tt_full[i, j, k] = (
                        tt_weights[0][:, i, :].dot(tt_weights[1][:, j, :])
                    ).dot(tt_weights[2][:, k, :]).squeeze()
        tt_full = jnp.array(tt_full)

        expected = phi.dot(tt_full.reshape(-1, order='F'))
        actual = predict_score_tt(self.x, kd, tt_weights, self.fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_prepare_buffer_tt(self):
        tt_weights, kd = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)
        buf = prepare_buffer_tt(self.x, kd, tt_weights, self.fmap)
        # Test shapes:
        expected = [(self.x.shape[0], r) for r in self.rank_list[1:-1]]
        actual = [b.shape for b in buf]
        self.assertTrue(np.allclose(actual, expected))
        # Test prediction ability:
        d = 0
        a_mtx = prepare_a_mtx(d, buf, self.fmap(self.x[:, d], None))
        wd_vec = tt_weights[d].reshape(-1, order='F')
        expected = predict_score_tt(self.x, kd, tt_weights, self.fmap)
        actual = a_mtx.dot(wd_vec)
        self.assertTrue(np.allclose(actual, expected))

    def test_update_weights_tt(self):
        tracker = LossTracker(
            self.x, self.y, self.beta_e, self.gamma_w, l2_loss_tt)
        w_tt, kd = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)
        buf = prepare_buffer_tt(self.x, kd, w_tt, self.fmap)
        tracker.track(w_tt, kd, self.fmap)
        _ = update_weights_tt(
            w_tt, kd, self.x, self.y, self.fmap, 
            self.gamma_w, self.beta_e, buf, tracker=tracker,
        )

        expected = np.sort(tracker.res_dict['loss'])[::-1]
        actual = tracker.res_dict['loss']
        self.assertTrue(np.allclose(actual, expected))

    def test_als_tt(self):
        tracker = LossTracker(
            self.x, self.y, self.beta_e, self.gamma_w, l2_loss_tt)
        w_tt, kd = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)
        _ = als_tt(
            w_tt, kd, self.x, self.y, self.fmap, 
            self.n_epoch, self.gamma_w, self.beta_e, tracker,
        )

        expected = np.sort(tracker.res_dict['loss'])[::-1]
        actual = tracker.res_dict['loss']
        self.assertTrue(np.allclose(actual, expected))

    def test_p_q_cores(self):
        w_tt, kd = init_weights_tt(
            self.m_order, self.rank_list, seed=self.seed)

        # Test prediction ability:
        d_core = 1
        k, q = divmod(d_core, kd) # q starts from zero -> for fmap
        wd_vec = w_tt[d_core].reshape(-1, order='F')
        fk_mtx = self.fmap(self.x[:, k], q)
        P = prepare_p_core_left(d_core, self.x, kd, w_tt, self.fmap)
        Q = prepare_q_core_right(d_core, self.x, kd, w_tt, self.fmap)
        a_mtx = prepare_a_mtx_v2(Q, fk_mtx, P)

        expected = predict_score_tt(self.x, kd, w_tt, self.fmap)
        actual = a_mtx.dot(wd_vec)
        self.assertTrue(np.allclose(actual, expected))

    def test_hess_one_core_tt(self):
        d_core = 1
        ind = sum(
            [
                self.rank_list[i]*self.m_order*self.rank_list[i+1] 
                for i in range(d_core)
            ]
        )
        shift = self.rank_list[d_core]*self.m_order*self.rank_list[d_core+1]
        H_jax = hess_full_jax(
            self.w_tt, self.kd, self.x, self.y, self.fmap, 
            self.gamma_w, self.beta_e, self.m_order, self.rank_list
        )

        expected = H_jax[ind:ind+shift, ind:ind+shift]
        actual = hess_one_core_tt(
            d_core, self.w_tt, self.kd, self.x, 
            self.fmap, self.gamma_w, self.beta_e
        )
        self.assertTrue(np.allclose(actual, expected))

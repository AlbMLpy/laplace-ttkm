import sys
import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, jacrev, hessian
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.trackers import LossTracker
from source.evaluation import l2_loss_tt
from source.matrix_operations import khatri_rao_row
from source.features import PPFeature, prepare_fmap
from source.model_functionality_tt import (
    als_tt,
    init_weights_tt,
    predict_score_tt,
    prepare_buffer_tt,
    update_weights_tt,
)

def prepare_a_mtx(ind: int, buf: list, fk_mtx):
    if ind == 0: 
        return khatri_rao_row(buf[ind], fk_mtx)
    elif ind == len(buf): 
        return khatri_rao_row(fk_mtx, buf[ind-1])
    else: 
        return khatri_rao_row(khatri_rao_row(buf[ind], fk_mtx), buf[ind-1])

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
        self.rank_list = [1, 4, 5, 1]
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
                
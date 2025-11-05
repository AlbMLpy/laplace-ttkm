import sys
import unittest

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.prob_functions import w_sample
from source.matrix_operations import tt2vec
from source.model_functionality_tt import (
    init_weights_tt, 
    get_tt_ind_shift,
)

class TestProbFunctions(unittest.TestCase):
    def setUp(self):
        self.m_order = 4
        self.tt_ranks = (1, 4, 5, 1)
        self.seed = 12
        self.w_tt, self.kd = init_weights_tt(
            self.m_order, self.tt_ranks, seed=self.seed
        )

    def test_w_sample(self):
        w_mean_vec = tt2vec(self.w_tt)
        ind, shift = get_tt_ind_shift(1, self.m_order, self.tt_ranks)
        w_sample_vec = w_sample(
            w_mean_vec, jnp.ones((shift,)*2), ind, shift, self.seed)

        self.assertTrue(
            np.abs((w_sample_vec - w_mean_vec)[:ind]).sum() == 0)
        self.assertTrue(
            np.abs((w_sample_vec-w_mean_vec)[ind:ind+shift]).sum() != 0)
        self.assertTrue(
            np.abs((w_sample_vec-w_mean_vec)[ind+shift:]).sum() == 0
        )

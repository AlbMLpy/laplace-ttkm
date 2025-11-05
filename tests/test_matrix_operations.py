import sys
import unittest

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.matrix_operations import (
    tt2vec,
    vec2tt,
    khatri_rao_row, 
)

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.w_tt = [
            jnp.array([[0, 1], [2, 3]])[None, ...], 
            jnp.array([[3, 4], [4, 5]])[..., None]
        ]
        self.w_vec = jnp.array([0, 2, 1, 3, 3, 4, 4, 5])

    def test_khatri_rao_row(self):
        # prepare the data:
        a = jnp.arange(1, 5).reshape(2, 2)
        b = jnp.arange(1, 7).reshape(2, 3)

        expected = jnp.array(
            [
                [ 1,  2,  3,  2,  4,  6],
                [12, 15, 18, 16, 20, 24],
            ]
        )
        actual = khatri_rao_row(a, b)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_tt2vec(self):
        expected = self.w_vec
        actual = tt2vec(self.w_tt)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_vec2tt(self):
        expected = self.w_tt
        actual = vec2tt(self.w_vec, 2, tuple([1, 2, 1]))
        self.assertTrue(
            np.all(
                [
                    np.allclose(actual[i], expected[i]) 
                    for i in range(len(actual))
                ]
            )
        )

    def test_tt2vec2tt(self):
        expected = tt2vec(self.w_tt)
        actual = tt2vec(vec2tt(tt2vec(self.w_tt), 2, tuple([1, 2, 1])))
        self.assertTrue(jnp.allclose(actual, expected))

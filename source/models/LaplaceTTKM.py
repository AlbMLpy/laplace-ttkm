from typing import Optional, Union

import jax.numpy as jnp
from sklearn.utils.validation import check_X_y

from .AbstractTNKM import AbstractTNKM
from ..features import Feature, PPFeature
from ..model_functionality_tt import (
    als_tt,
    init_weights_tt,
    predict_score_tt,
    hess_cov_estimation,
)

class LaplaceTTKM(AbstractTNKM):
    """
    Bayesian Tensor Network Kernel Machine (LA-TNKM) that 
    uses a (linearized) Laplace approximation for Bayesian inference.

    Currently, the implementation is restricted to using 
    the tensor train network (TT).

    Parameters
    ----------
    tt_ranks : tuple, default=None
        TT ranks of the TT weights tensor.

    fmap : Feature, default=PPFeature()
        Nonlinear data mapping to get new features. 
        Other options can be: [PPNFeature, RBFFeature].

    m_order : int, default=2
        The number of newly generated features per data feature, x_d.
        E.g., in case of 'PPFeature', m_order is the order of the polynomial.

    n_epoch : int, default=1
        The number of full ALS updates (sweeps).

    beta_e : float, optional, default=1.0
        Noise precision hyperparameter. If 'beta_e=None' the hyperparameter 
        is evaluated with variational inference.

    gamma_w : float, optional, default=1e-3
        Prior precision hyperparameter. If 'gamma_w=None' the hyperparameter 
        is evaluated with variational inference. Note that, this hyperparameter 
        is sensitive and can lead to zero solution, w_ten = 0, very quickly.

    pd_mode : str, default='lla'
        Type of the predictive distribution: 'lla' refers to Linearized Laplace
        Approximation (LLA), 'la' refers to Laplace Approximation (LA). 

    hess_type : str or int, optional, default='mf'
        Type of Hessian approximation: 'full' represents full Hessian mode, 
        'gauss_newton' represents generalized Gauss-Newton, 'block' refers to 
        block-diagonal approximation, 'mf' represents diagonal approximation 
        and 'integer' means being Bayesian only to one TT core (starts with 0).

    hess_th : float, optional, default=None
        Thresholding hyperparameter to eliminate problematic eigenvalues of the 
        Hessian or its approximation. If 'hess_th=None' then no thresholding 
        is applied.

    seed : int, optional, default=None
        Determines random number generation used to initialize 
        the model parameters. Pass an int for reproducible results.

    opt_params : dict, optional, default=None
        This dict defines the optimizer used and its hyperparameters.
        If 'opt_params=None' then ALS is used.

    n_epoch_vi : int, default=1
        The number of variational inference updates of 
        the model parameters: w_ten, beta_e, gamma_w.

    pd_samples : int, default=30
        The number samples to estimate the expectation
        (predictive distribution) with Monte Carlo sampling.

    pd_sample_seed : int, optional, default=None
        Determines random number generation used by Monte Carlo sampler to
        estimate the expectation. Pass an int for reproducible results.

    beta_e_samples : int, default=10
        Determines the number of samples used to estimate the noise precision.

    tracker : Tracker object, optional, default=None
        This object can be used to gather useful statistics during training.
        If 'tracker=None' then no tracking is used. 

    Attributes
    ----------
    w_mean : list[array-like]
        List of Arrays containing the TT weights.

    w_cholesky : array-like
        Array containing Cholesky factor matrix of the corresponding 
        covariance matrix (inverse of the Hessian).

    Examples
    --------
    In Progress...
    
    """

    def __init__(
        self, 
        tt_ranks: Optional[tuple] = None, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        beta_e: Optional[float] = 1.0,
        gamma_w: Optional[float] = 1e-3,
        pd_mode: str = 'la', 
        hess_type: Optional[Union[str, int]] = 'mf',
        hess_th: Optional[float] = None,
        seed: Optional[int] = None,
        opt_params: Optional[dict] = None,
        n_epoch_vi: int = 1,
        pd_samples: int = 30, 
        beta_e_samples: int = 10, 
        tracker: Optional[object] = None,
    ):
        super().__init__(
            fmap, m_order, n_epoch, beta_e, gamma_w, pd_mode, 
            hess_type, hess_th, seed, opt_params, n_epoch_vi,
            pd_samples, beta_e_samples, tracker,
        )
        self.tt_ranks = tt_ranks 
        self._qbase = None
        self._init_type = None

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        """
        Fit Bayesian tensor train kernel machine.

        Parameters
        ----------
        X : array-like of shape (n_samples, d_dim)
            Training data matrix.

        y : array-like of shape (n_samples,)
            Target values.

        xy_test : tuple, optional, default=None
            Test dataset.

        Returns
        -------
        self : object
            LaplaceTTKM class instance.
        """
        # Data checks:
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        # Init model parameters:
        self.w_mean, self.kd = init_weights_tt(
            self.m_order, 
            self.tt_ranks, 
            self._qbase, 
            self._init_type, 
            self.seed, 
            self._dtype
        )
        # VI training loop:
        for _ in range(self.n_epoch_vi):
            self._update_w(X, y, xy_test) # Update weights mean and cov.;
            if not self.upd_beta_e and not self.upd_gamma_w:
                break 
            if self.upd_gamma_w:
                self._update_gamma_w() # Update weights prior precision; 
            if self.upd_beta_e:
                self._update_beta_e(X, y) # Update noise precision;
        self.is_fitted_ = True
        return self
    
    def _update_w(self, X, y, xy_test: Optional[tuple] = None):
        """ 
        Update TT weights (mean) and covariance matrix (Cholesky factor). 
        """
        if xy_test: raise NotImplementedError()

        self.w_mean = als_tt(
            self.w_mean, self.kd, X, y, self._fmap, self.n_epoch, 
            self.gamma_w, self.beta_e, self.tracker,
        )
        #self.w_ten, self.w_shape = process_weights(self.w_ten)
        self.w_hess, self.w_cov, self.w_cholesky = hess_cov_estimation(
            self.w_mean, self.kd, X, self._fmap, self.gamma_w, 
            self.beta_e, self.hess_type, self.hess_th,
        )

    def _predict_mean(self, X):
        """ Compute mean of the predictive distribution. """
        return predict_score_tt(X, self.kd, self.w_mean, self._fmap)

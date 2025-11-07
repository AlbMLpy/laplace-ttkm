from collections import defaultdict

from .evaluation import norm_frob
from .matrix_operations import tt2vec
from .model_functionality_tt import get_tt_ranks
from .general_functions import update_results_dict

class LossTracker:
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss):
        self.res_dict = defaultdict(list)
        self.x_train = x_train
        self.y_train = y_train
        self.beta_e = beta_e
        self.gamma_w = gamma_w
        self.loss = loss

    def track(self, w_tt, kd, fmap, beta_e=None, gamma_w=None):
        _beta_e = beta_e if beta_e else self.beta_e
        _gamma_w = gamma_w if gamma_w else self.gamma_w

        train_loss = self.loss(
            w_tt, kd, self.x_train, self.y_train, 
            fmap, _gamma_w, _beta_e
        ).item()
        update_results_dict(self.res_dict, loss=train_loss)

class GradTracker(LossTracker):
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss, grad_w):
        super().__init__(x_train, y_train, beta_e, gamma_w, loss)
        self.grad_w = grad_w

    def track(self, w_tt, kd, fmap, beta_e=None, gamma_w=None):
        _beta_e = beta_e if beta_e else self.beta_e
        _gamma_w = gamma_w if gamma_w else self.gamma_w

        w_vec = tt2vec(w_tt)
        m_order, rank_list = w_tt[0].shape[1], get_tt_ranks(w_tt)
        train_loss = self.loss(
            w_tt, kd, self.x_train, self.y_train, 
            fmap, _gamma_w, _beta_e
        ).item()
        grad_norm = norm_frob(
            self.grad_w(
                w_vec, kd, self.x_train, self.y_train, 
                fmap, _gamma_w, _beta_e, m_order, rank_list,
            )
        ).item()
        update_results_dict(
            self.res_dict, loss=train_loss, grad_norm=grad_norm,
        )

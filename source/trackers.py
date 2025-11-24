from collections import defaultdict

import pandas as pd

from .matrix_operations import tt2vec
from .evaluation import norm_frob, rmse, nll
from .model_functionality_tt import get_tt_ranks, predict_score_tt
from .general_functions import update_results_dict, extend_results_dict

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
        update_results_dict(
            self.res_dict, loss=train_loss,
            beta_e=_beta_e, gamma_w=_gamma_w,
        )

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
            beta_e=_beta_e, gamma_w=_gamma_w,
        )

class LossMetricsTracker(LossTracker):
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss, xy_test):
        super().__init__(x_train, y_train, beta_e, gamma_w, loss)
        self.x_test, self.y_test = xy_test

    def track(self, w_tt, kd, fmap, beta_e=None, gamma_w=None):
        _b = beta_e if beta_e else self.beta_e
        _g = gamma_w if gamma_w else self.gamma_w
        train_loss = self.loss(
            w_tt, kd, self.x_train, self.y_train, fmap, _g, _b).item()
        test_loss = self.loss(
            w_tt, kd, self.x_test, self.y_test, fmap, _g, _b).item()
        ys_train = predict_score_tt(self.x_train, kd, w_tt, fmap)
        ys_test = predict_score_tt(self.x_test, kd, w_tt, fmap)
        update_results_dict(
            self.res_dict, train_loss=train_loss, test_loss=test_loss,
            rmse_train=rmse(ys_train, self.y_train), 
            rmse_test=rmse(ys_test, self.y_test),
            beta_e=_b, gamma_w=_g,
        )

class CVTracker:
    def __init__(self, res_dir, res_name, cv_name):
        self.res_dict = defaultdict(list)
        self.extra_dict = defaultdict(list)
        self.res_dir = res_dir
        self.res_name = res_name
        self.cv_name = cv_name

    def track(self, trial, x_train, x_test, y_train, y_test, model, model_info):
        ys_train, ys_std_train = model.predict(x_train, return_std=True)
        ys_test, ys_std_test = model.predict(x_test, return_std=True)

        update_results_dict(
            self.res_dict, 
            trial=trial, 
            rmse_train=rmse(y_train, ys_train),
            rmse_test=rmse(y_test, ys_test),
            nll_train=nll(ys_train, ys_std_train**2, y_train),
            nll_test=nll(ys_test, ys_std_test**2, y_test),
            train_time=model_info.pop('train_time'),
            gamma_w=model['model'].gamma_w,
            beta_e=model['model'].beta_e,
        )
        if model_info:
            extend_results_dict(
                self.extra_dict,
                trial=[trial,]*len(model_info['scores']),
                **model_info,
            )

    def save(self):
        pd.DataFrame(
            self.res_dict).to_csv(self.res_dir / f'{self.res_name}.csv')
        pd.DataFrame(
            self.extra_dict).to_csv(self.res_dir / f'{self.cv_name}.csv')

    def load(self):
        res_df = pd.read_csv(
            self.res_dir / f'{self.res_name}.csv', index_col=0)
        cv_df = pd.read_csv(
            self.res_dir / f'{self.cv_name}.csv', index_col=0)
        return res_df, cv_df

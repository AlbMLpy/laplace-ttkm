from collections import defaultdict

from .general_functions import update_results_dict

class LossTracker:
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss):
        self.res_dict = defaultdict(list)
        self.x_train = x_train
        self.y_train = y_train
        self.beta_e = beta_e
        self.gamma_w = gamma_w
        self.loss = loss

    def track(self, w_tt, kd, fmap):
        train_loss = self.loss(
            w_tt, kd, self.x_train, self.y_train, 
            fmap, self.gamma_w, self.beta_e
        )
        update_results_dict(
            self.res_dict, loss=train_loss
        )

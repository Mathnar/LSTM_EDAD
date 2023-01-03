import torch
import torch.nn as nn


def MSEAUC_approx(x, x_hat, y, lambda_auc):

    # Computing error for each row
    err = torch.pow(x - x_hat, 2).mean(axis = (1, 2))

    # Selecting error of positive and negative example
    err_n = err[y == 1]
    err_a = err[y > 1]
    n_a = (err_a.shape)[0]
    n_n = (err_n.shape)[0]

    # If there are positive examples compute the AUC penalty
    if n_a > 0:
        diff = err_a.view(-1, 1).unsqueeze(1) - err_n.view(-1, 1)
        exp = torch.sigmoid(diff).sum()
        auc = lambda_auc * exp / (n_a * n_n)
        mean_loss = err.mean()
        penalized_loss = mean_loss + auc
        return penalized_loss, mean_loss
    else:
        mean_loss = err.mean()
        return mean_loss

class MSEAUCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = MSEAUC_approx

    def forward(self, x_hat, x_true, y, lambda_auc):
        loss = self.loss(x_hat, x_true, y, lambda_auc)
        return loss


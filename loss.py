import torch
from torch import nn


class AutoRecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, r_hat, mask_r, model):
        residual = torch.multiply(r - r_hat, mask_r)
        cost = torch.square(torch.norm(residual))
        return cost + model.regularization()

import torch
from torch import nn, square, div, Tensor
from torch.linalg import norm
import pytorch_lightning as pl
from loss import AutoRecLoss


class AutoRec(nn.Module):
    def __init__(self, d: int = 100, k: int = 10, weight_decay: int = 10, min_rating: int = 1, max_rating: int = 5):
        """
            d: dimension of input and output
            k: dimension of latent
            weight_decay: regularization parameter
        """
        super().__init__()
        self.W = nn.Parameter(torch.rand(d, k))
        self.b = nn.Parameter(torch.rand(d))
        self.V = nn.Parameter(torch.rand(k, d))
        self.mu = nn.Parameter(torch.rand(k))
        
        self.weight_decay = weight_decay
        self.min_rating = 1
        self.max_rating = 5

    def regularization(self) -> Tensor:
        """
            loss = criterion(..., ...) + model.regularization()
        """
        return div(self.weight_decay, 2) * (square(norm(self.W)) + square(norm(self.V)))

    def forward(self, r) -> Tensor:
        """
            r: (batch, d)
        """
        pre_encoder = self.V.matmul(r.T).T + self.mu
        encoder = torch.sigmoid(pre_encoder)

        pre_decoder = self.W.matmul(encoder.T).T + self.b
        decoder = torch.clip(pre_decoder, self.min_rating, self.max_rating)

        return decoder


class AutoRecModule(pl.LightningModule):
    def __init__(
        self, d: int = 100, k: int = 10, weight_decay: int = 10, min_rating: int = 1, max_rating: int = 5, optimizer: dict = None
    ):
        super().__init__()
        self.autorec = AutoRec(d, k, weight_decay, min_rating, max_rating)
        self.criterion = AutoRecLoss()

        self.cfg_optimizer = optimizer

    def forward(self, r) -> Tensor:
        return self.autorec(r)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.cfg_optimizer)
        return optimizer

    def training_step(self, batch, batch_idx):
        r, mask_r = batch
        r_hat = self.autorec(r)
        loss = self.criterion(r, r_hat, mask_r, self.autorec)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        r, mask_r = batch
        r_hat = self.autorec(r)

        loss = self.criterion(r, r_hat, mask_r, self.autorec)
        rmse = self.cal_rmse(r, r_hat, mask_r)

        print(rmse.item())

        self.log("val_loss", loss)
        self.log("val_rmse", rmse)

        return loss, rmse

    def test_step(self, batch, batch_idx):
        r, mask_r = batch
        r_hat = self.autorec(r)
        loss = self.criterion(r, r_hat, mask_r, self.autorec)
        rmse = self.cal_rmse(r, r_hat, mask_r)

        self.log("test_loss", loss)
        self.log("test_rmse", rmse)

        return loss, rmse

    def cal_rmse(self, r, r_hat, mask_r):
        r_hat = torch.multiply(r_hat, mask_r)
        mse_loss = nn.functional.mse_loss(r, r_hat)
        return torch.sqrt(mse_loss)

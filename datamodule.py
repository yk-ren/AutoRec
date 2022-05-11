import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class MovielenDataModule(pl.LightningDataModule):
    def __init__(self, trainset, testset, batch_size):
        super().__init__()
        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, pin_memory=True, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=len(self.testset), pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.testset, batch_size=len(self.testset), pin_memory=True
        )

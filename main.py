import argparse
import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from dataset import MovielenDataset
from datamodule import MovielenDataModule
from model import AutoRecModule

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config path")
    parser.add_argument("-cp", help="config path")  # config path
    parser.add_argument("-cn", help="config name")  # config name

    args = parser.parse_args()

    @hydra.main(config_path=args.cp, config_name=args.cn)
    def main(cfg: DictConfig):
        trainset = MovielenDataset(subset="train", **cfg.dataset)
        testset = MovielenDataset(subset="test", **cfg.dataset)

        dm = MovielenDataModule(trainset, testset, **cfg.datamodule)

        model = AutoRecModule(**cfg.model)

        logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.logger)

        trainer = pl.Trainer(logger=logger, **cfg.trainer)

        if cfg.ckpt.have_ckpt:
            trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            trainer.fit(model, datamodule=dm)

        if cfg.ckpt.have_ckpt:
            trainer.test(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            trainer.test(model, datamodule=dm)

    main()

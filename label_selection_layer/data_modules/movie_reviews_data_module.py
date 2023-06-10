from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PreprocessedMovieReviewsDataset(Dataset):
    def __init__(self, datadir: Path, mode: str, target: Optional[str] = None):
        super().__init__()
        self.mode = mode
        self.datadir = datadir

        if mode == "train":
            self.X = np.load(self.datadir.joinpath("train_preprocessed_X.npy")).astype(
                np.int32
            )

            if target == "gt":
                self.y = np.load(self.datadir.joinpath("targets_train.npy")).astype(
                    np.float32
                )

            elif target == "mean":
                self.y = np.load(
                    self.datadir.joinpath("aggregated_targets/targets_by_mean.npy")
                ).astype(np.float32)

            elif target == "ds":
                self.y = np.load(
                    self.datadir.joinpath("aggregated_targets/targets_by_DS.npy")
                ).astype(np.float32)

            elif target == "annotation":
                self.y = np.load(self.datadir.joinpath("train_answers.npy")).astype(
                    np.float32
                )

            else:
                raise ValueError

        elif mode == "test":
            # test data
            self.X = np.load(self.datadir.joinpath("test_preprocessed_X.npy")).astype(
                np.int32
            )
            self.y = np.load(self.datadir.joinpath("targets_test.npy")).astype(
                np.float32
            )

        else:
            raise ValueError

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        return X, y


class PreprocessedMovieReviewsDataModule(pl.LightningDataModule):
    def __init__(self, datadir: Path, target: str, batch_size: int = 32):
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = batch_size
        self.datadir = datadir
        self.target = target

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = PreprocessedMovieReviewsDataset(
            datadir=self.datadir, mode="train", target=self.target
        )
        self.valid_data = None
        self.test_data = PreprocessedMovieReviewsDataset(
            datadir=self.datadir, mode="test"
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

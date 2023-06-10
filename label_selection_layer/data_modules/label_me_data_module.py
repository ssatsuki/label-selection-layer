from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PreprocessedLabelMeDataset(Dataset):
    def __init__(self, datadir: Path, mode: str, label: Optional[str] = None):
        super().__init__()
        self.mode = mode
        self.datadir = datadir

        if mode == "train":
            self.X = np.load(self.datadir.joinpath("data_train_vgg16.npy"))

            if label == "gt":
                self.y = np.load(self.datadir.joinpath("labels_train.npy"))

            elif label == "mv":
                self.y = np.load(self.datadir.joinpath("labels_train_mv.npy"))

            elif label == "ds":
                self.y = np.load(self.datadir.joinpath("labels_train_DS.npy"))

            elif label == "glad":
                self.y = np.load(self.datadir.joinpath("labels_train_glad.npy"))

            elif label == "annotation":
                self.y = np.load(self.datadir.joinpath("answers.npy"))

            else:
                raise ValueError

        elif mode == "valid":
            # validation data
            self.X = np.load(self.datadir.joinpath("data_valid_vgg16.npy"))
            self.y = np.load(self.datadir.joinpath("labels_valid.npy"))

        elif mode == "test":
            # test data
            self.X = np.load(self.datadir.joinpath("data_test_vgg16.npy"))
            self.y = np.load(self.datadir.joinpath("labels_test.npy"))

        else:
            raise ValueError

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        return X, y


class PreprocessedLabelMeDataModule(pl.LightningDataModule):
    def __init__(self, datadir: Path, label: str, batch_size: int = 32):
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = batch_size
        self.datadir = datadir
        self.label = label

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = PreprocessedLabelMeDataset(
            datadir=self.datadir, mode="train", label=self.label
        )
        self.valid_data = PreprocessedLabelMeDataset(datadir=self.datadir, mode="valid")
        self.test_data = PreprocessedLabelMeDataset(datadir=self.datadir, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PreprocessedNERMTurkDataset(Dataset):
    def __init__(self, datadir: Path, mode: str, label: Optional[str] = None):
        super().__init__()
        self.mode = mode
        self.datadir = datadir

        if mode == "train":
            self.X = np.load(
                self.datadir.joinpath("ner_mturk_encoded_train_features.npy")
            )

            if label == "gt":
                self.y = np.load(self.datadir.joinpath("ner_mturk_train_labels_gt.npy"))

            elif label == "mv":
                self.y = np.load(self.datadir.joinpath("ner_mturk_train_labels_mv.npy"))

            elif label == "ds":
                self.y = np.load(self.datadir.joinpath("ner_mturk_train_labels_ds.npy"))

            elif label == "annotation":
                self.y = np.load(
                    self.datadir.joinpath("ner_mturk_train_labels_answers.npy")
                )

            else:
                raise ValueError

        elif mode == "valid":
            self.X = np.load(
                self.datadir.joinpath("ner_mturk_encoded_valid_features.npy")
            )
            self.y = np.load(self.datadir.joinpath("ner_mturk_valid_labels_gt.npy"))

        elif mode == "test":
            # test data
            self.X = np.load(
                self.datadir.joinpath("ner_mturk_encoded_test_features.npy")
            )
            self.y = np.load(self.datadir.joinpath("ner_mturk_test_labels.npy"))

        else:
            raise ValueError

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        return X, y


class PreprocessedNERMTurkDataModule(pl.LightningDataModule):
    def __init__(
        self, datadir: Path, label: str, batch_size: int = 64, num_workers: int = 0
    ):
        super().__init__()
        self.prepare_data_per_node = False
        self.batch_size = batch_size
        self.label = label
        self.datadir = datadir
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_data = PreprocessedNERMTurkDataset(
            datadir=self.datadir, mode="train", label=self.label
        )
        self.valid_data = PreprocessedNERMTurkDataset(
            datadir=self.datadir, mode="valid"
        )
        self.test_data = PreprocessedNERMTurkDataset(datadir=self.datadir, mode="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

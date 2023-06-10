from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ...nn_modules import CrowdRegressionLayer


class BasicModel(pl.LightningModule):
    def __init__(
        self,
        num_embeddings: int,
        embeddings: np.ndarray,
        dropout_ratio: float = 0.5,
        learning_rate: float = 1e-3,
        embedding_dim: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embeddings"])

        self.encode = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.model = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(p=self.hparams.dropout_ratio),
            nn.Conv1d(128, 128, 5),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten(),
            nn.Linear(128 * 39, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.encode(x)
        z = z.transpose(1, 2)
        z = self.model(z)
        z = z.squeeze()
        return z

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self(x)
        train_loss = F.mse_loss(z, y)

        logs = {"train_loss": train_loss}
        batch_dictionary = {
            "loss": train_loss,
            "log": logs,
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        val_loss = F.mse_loss(z, y)

        logs = {"val_loss": val_loss}
        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        test_loss = F.mse_loss(z, y)

        logs = {"test_loss": test_loss}
        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class CrowdLayerModel(pl.LightningModule):
    def __init__(
        self,
        num_embeddings: int,
        embeddings: np.ndarray,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        dropout_ratio: float = 0.5,
        crowd_layer_mode: str = "S",
        embedding_dim: int = 300,
    ):
        if n_workers is None:
            raise ValueError("must set n_workers.")

        super().__init__()
        self.save_hyperparameters()

        self.encode = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.model = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(p=self.hparams.dropout_ratio),
            nn.Conv1d(128, 128, 5),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten(),
            nn.Linear(128 * 39, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.crowd_layer = CrowdRegressionLayer(
            self.hparams.n_workers,
            mode=self.hparams.crowd_layer_mode,
        )

    def forward(self, x):
        z = self.encode(x)
        z = z.transpose(1, 2)
        z = self.model(z)
        z = z.squeeze()
        return z

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self(x)
        z = self.crowd_layer(z)

        y_ = y.view(-1)
        z_ = z.transpose(0, 1).contiguous().view(-1)

        mse = F.mse_loss(z_, y_, reduction="none")
        mse = torch.where(y_ != 999999999, mse, 0)
        train_loss = mse.sum() / (y_ != 999999999).sum()
        logs = {"train_loss": train_loss}

        batch_dictionary = {
            "loss": train_loss,
            "log": logs,
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        val_loss = F.mse_loss(z, y)
        logs = {"val_loss": val_loss}

        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        test_loss = F.mse_loss(z, y)
        logs = {"test_loss": test_loss}

        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class LabelSelectionModel(pl.LightningModule):
    def __init__(
        self,
        num_embeddings: int,
        embeddings: np.ndarray,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        dropout_ratio: float = 0.5,
        c: float = 0.8,
        d: int = 3,
        lambda_: float = 32.0,
        alpha: float = 0.5,
        embedding_dim: int = 300,
        selective_mode: str = "simple",
    ):
        if n_workers is None:
            raise ValueError("must set n_workers.")

        super().__init__()
        self.save_hyperparameters()

        self.encode = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.features = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(p=self.hparams.dropout_ratio),
            nn.Conv1d(128, 128, 5),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten(),
            nn.Linear(128 * 39, 32),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            self.features,
            nn.Linear(32, 1),
        )

        if self.hparams.selective_mode == "simple":
            self.W = nn.Parameter(torch.Tensor(np.zeros(self.hparams.n_workers)))

        elif self.hparams.selective_mode == "target-wise":
            self.W = nn.Parameter(torch.Tensor(np.ones(self.hparams.n_workers)))
            self.bias = nn.Parameter(torch.Tensor(np.ones(self.hparams.n_workers)))

        elif self.hparams.selective_mode == "feature-based":
            self.selector = nn.Sequential(
                self.features,
                nn.Linear(32, self.hparams.n_workers),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f"'{self.hparams.selective_mode}' is unsupported as a selective_mode."
            )

        self.selection_layer = nn.Sigmoid()

    def forward(self, x):
        z = self.encode(x)
        z = z.transpose(1, 2)
        z = self.model(z)
        z = z.squeeze()
        return z

    def selection_prob(self, x, y):
        if self.hparams.selective_mode == "simple":
            return self.selection_layer(self.W)

        elif self.hparams.selective_mode == "target-wise":
            return self.selection_layer((self.W * y + self.bias) ** self.hparams.d)

        # NOTE: feature-based case
        z = self.encode(x)
        z = z.transpose(1, 2)
        return self.selector(z)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self(x)
        s = self.selection_prob(x, y)

        batch_size = y.shape[0]

        z_ = z.repeat(135, 1).transpose(0, 1).contiguous().view(-1)

        if self.hparams.selective_mode == "simple":
            y_ = y.view(-1)
            s_ = s.repeat(batch_size, 1).view(-1)

        elif self.hparams.selective_mode in {"target-wise", "feature-based"}:
            y_ = y.view(-1)
            s_ = s.view(-1)

        else:
            raise ValueError

        ell = F.mse_loss(z_, y_, reduction="none")
        ell = torch.where(y_ != 999999999, ell, 0)
        phi = s_[y_ != 999999999].mean()
        Am_size = (y_ != 999999999).sum()
        L = (ell.mul(s_).sum() / Am_size) / phi
        psi = torch.pow(torch.max(torch.tensor(0), self.hparams.c - phi), 2)
        train_loss = L + self.hparams.lambda_ * psi

        logs = {"train_loss": train_loss}

        batch_dictionary = {
            "loss": train_loss,
            "phi": phi,
            "psi": psi,
            "log": logs,
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)
        self.log(
            "phi/Train",
            torch.stack([x["phi"] for x in outputs]).mean(),
            on_epoch=True,
            logger=True,
        )
        self.log(
            "psi/Train",
            torch.stack([x["psi"] for x in outputs]).mean(),
            on_epoch=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        val_loss = F.mse_loss(z, y)
        logs = {"val_loss": val_loss}

        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        test_loss = F.mse_loss(z, y)
        logs = {"test_loss": test_loss}

        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

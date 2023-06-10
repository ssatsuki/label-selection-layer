from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ...nn_modules import CrowdLayer


class BasicModel(pl.LightningModule):
    def __init__(
        self,
        n_labels: Optional[int] = None,
        learning_rate: float = 1e-3,
        middle_layer_dim: int = 128,
        dropout_ratio: float = 0.5,
    ):
        if n_labels is None:
            raise ValueError("must set n_labels.")

        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(4 * 4 * 512, self.hparams.middle_layer_dim),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout_ratio),
            nn.Linear(self.hparams.middle_layer_dim, self.hparams.n_labels),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.model(x)
        train_loss = F.cross_entropy(z, y)
        correct = z.argmax(dim=1).eq(y).sum().item()
        logs = {"train_loss": train_loss}
        batch_dictionary = {
            "loss": train_loss,
            "log": logs,
            "correct": correct,
            "total": len(y),
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Train", correct / total, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        val_loss = F.cross_entropy(z, y)
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        logs = {"val_loss": val_loss}
        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Validation", correct / total, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        test_loss = F.cross_entropy(z, y)
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        logs = {"test_loss": test_loss}
        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Test", correct / total, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        z = self(x)
        z = F.softmax(z, dim=1)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class CrowdLayerModel(pl.LightningModule):
    def __init__(
        self,
        n_labels: Optional[int] = None,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        middle_layer_dim: int = 128,
        dropout_ratio: float = 0.5,
        crowd_layer_mode: str = "MW",
    ):
        if (n_labels is None) or (n_workers is None):
            raise ValueError("must set both of n_labels and n_workers.")

        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(4 * 4 * 512, self.hparams.middle_layer_dim),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout_ratio),
            nn.Linear(self.hparams.middle_layer_dim, self.hparams.n_labels),
        )

        self.crowd_layer = CrowdLayer(
            self.hparams.n_labels,
            self.hparams.n_workers,
            mode=self.hparams.crowd_layer_mode,
        )

    def forward(self, x):
        return self.model(x)

    def forward_in_train_step(self, x):
        z = self(x)
        z = F.softmax(z, dim=1)
        z = self.crowd_layer(z)
        return z

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.forward_in_train_step(x)

        z_ = z.view(-1, self.hparams.n_labels)
        y_ = y.transpose(0, 1).contiguous().view(-1).type(torch.int64)

        train_loss = F.cross_entropy(z_, y_, ignore_index=-1)
        correct = z_.argmax(dim=1).eq(y_).sum().item()

        logs = {"train_loss": train_loss}

        batch_dictionary = {
            "loss": train_loss,
            "log": logs,
            "correct": correct,
            "total": (y != -1).sum(),
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Train", correct / total, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        val_loss = F.cross_entropy(z, y)
        logs = {"val_loss": val_loss}
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Validation", correct / total, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        test_loss = F.cross_entropy(z, y)
        logs = {"test_loss": test_loss}
        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Test", correct / total, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        z = self(x)
        z = F.softmax(z, dim=1)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class LabelSelectionModel(pl.LightningModule):
    def __init__(
        self,
        n_labels: Optional[int] = None,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        middle_layer_dim: int = 128,
        dropout_ratio: float = 0.5,
        c: float = 0.8,
        lambda_: float = 32.0,
        alpha: float = 0.5,
        selective_mode: str = "simple",
    ):
        if (n_labels is None) or (n_workers is None):
            raise ValueError("must set both of n_labels and n_workers.")

        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            torch.nn.Flatten(),
            nn.Linear(4 * 4 * 512, self.hparams.middle_layer_dim),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout_ratio),
        )

        self.model = nn.Sequential(
            self.features,
            nn.Linear(self.hparams.middle_layer_dim, self.hparams.n_labels),
        )

        if self.hparams.selective_mode == "simple":
            self.W = nn.Parameter(torch.Tensor(np.zeros(self.hparams.n_workers)))

        elif self.hparams.selective_mode == "classwise":
            self.W = nn.Parameter(
                torch.Tensor(np.zeros((self.hparams.n_labels, self.hparams.n_workers)))
            )

        elif self.hparams.selective_mode == "feature-based":
            self.selector = nn.Sequential(
                self.features,
                nn.Linear(self.hparams.middle_layer_dim, self.hparams.n_workers),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f"'{self.hparams.selective_mode}' is unsupported as a selective_mode."
            )

        self.selection_layer = nn.Sigmoid()

    def forward(self, x):
        z = self.model(x)
        return z

    def selection_prob(self, x=None):
        if self.hparams.selective_mode in {"simple", "classwise"}:
            return self.selection_layer(self.W)

        # NOTE: feature-based case
        return self.selector(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self(x)
        s = self.selection_prob(x)

        batch_size = y.shape[0]

        z_ = (
            z.repeat(self.hparams.n_workers, 1, 1)
            .transpose(0, 1)
            .contiguous()
            .view(-1, self.hparams.n_labels)
        )

        if self.hparams.selective_mode == "simple":
            y_ = y.view(-1)
            s_ = s.repeat(batch_size, 1).view(-1)

        elif self.hparams.selective_mode == "classwise":
            y = y.unsqueeze(2)
            y_ = y.view(-1)
            s_ = torch.where(
                y != -1,
                torch.gather(
                    input=s.repeat(batch_size, 1, 1).transpose(1, 2),
                    dim=2,
                    index=torch.where(y != -1, y, 0),
                ),
                0,
            ).view(-1)

        elif self.hparams.selective_mode == "feature-based":
            y_ = y.view(-1)
            s_ = s.view(-1)

        else:
            raise ValueError

        ell = F.cross_entropy(z_, y_, ignore_index=-1, reduction="none")
        phi = s_[y_ != -1].mean()
        Am_size = (y_ != -1).sum()
        L = (ell.mul(s_).sum() / Am_size) / phi
        psi = torch.pow(torch.max(torch.tensor(0), self.hparams.c - phi), 2)
        train_loss = L + self.hparams.lambda_ * psi

        logs = {"train_loss": train_loss}

        batch_dictionary = {
            "loss": train_loss,
            "phi": phi,
            "psi": psi,
            "log": logs,
            "total": (y != -1).sum(),
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
        val_loss = F.cross_entropy(z, y)
        logs = {"val_loss": val_loss}

        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        batch_dictionary = {
            "loss": val_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Validation", correct / total, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        test_loss = F.cross_entropy(z, y)
        logs = {"test_loss": test_loss}

        correct = z.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        batch_dictionary = {
            "loss": test_loss,
            "log": logs,
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        self.log("Loss/Test", avg_loss, on_epoch=True, logger=True)
        self.log("Accuracy/Test", correct / total, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        z = self(x)
        z = F.softmax(z, dim=1)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

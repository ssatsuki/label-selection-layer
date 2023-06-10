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
        num_embeddings: int,
        embeddings: np.ndarray,
        dropout_ratio: float = 0.5,
        n_labels: Optional[int] = None,
        learning_rate: float = 1e-3,
        embedding_dim: int = 300,
        max_sequence_length: int = 109,
    ):
        if n_labels is None:
            raise ValueError("must set n_labels.")

        super().__init__()
        self.save_hyperparameters(ignore=["embeddings"])

        self.encode = nn.Sequential(
            nn.Embedding(num_embeddings, embedding_dim).from_pretrained(
                embeddings, freeze=False
            ),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=512, kernel_size=5, padding="same"),
        )

        self.gru = nn.Sequential(
            nn.GRU(input_size=512, hidden_size=50, batch_first=True),
        )

        self.outputs = nn.Sequential(
            nn.Linear(in_features=50, out_features=n_labels),
        )

    def forward(self, x):
        z = self.encode(x)
        z = z.transpose(2, 1)
        z = self.conv(z)
        z = z.transpose(2, 1)
        z = self.gru(z)[0]
        z = self.outputs(z)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x).view(-1, 10)
        loss = F.cross_entropy(z, y.view(-1, 10).argmax(axis=1))
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        z = self(x).view(-1, 10)
        loss = F.cross_entropy(z, y.view(-1, 10).argmax(axis=1))
        return {"loss": loss, "log": {"val_loss": loss}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("Loss/Train", avg_loss, on_epoch=True, logger=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.current_epoch == 0:
            self.log("Loss/Validation", 1e-1, on_epoch=True, logger=True)
        else:
            self.log("Loss/Validation", avg_loss, on_epoch=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        z = self(x)
        return F.softmax(z, dim=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class CrowdLayerModel(BasicModel):
    def __init__(
        self,
        num_embeddings: int,
        embeddings: np.ndarray,
        dropout_ratio: float = 0.5,
        n_labels: Optional[int] = None,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        embedding_dim: int = 300,
        max_sequence_length: int = 109,
        crowd_layer_mode: str = "MW",
    ):
        if n_labels is None:
            raise ValueError("must set n_labels.")

        if n_workers is None:
            raise ValueError("must set n_workers.")

        super().__init__(
            num_embeddings=num_embeddings,
            embeddings=embeddings,
            dropout_ratio=dropout_ratio,
            n_labels=n_labels,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
        )
        self.save_hyperparameters(ignore=["embeddings"])

        self.softmax = nn.Softmax(dim=2)

        self.crowd_layer = CrowdLayer(
            self.hparams["n_labels"],
            self.hparams["n_workers"],
            mode=self.hparams["crowd_layer_mode"],
        )

    def forward_in_train_step(self, x):
        z = self(x)
        z = self.softmax(z)
        if self.hparams["crowd_layer_mode"] == "MW":
            z = z.view(
                -1, 1, self.hparams["max_sequence_length"], self.hparams["n_labels"]
            )
            z = self.crowd_layer(z)
        elif self.hparams["crowd_layer_mode"] in {"VW", "VB", "VW+B"}:
            z = self.crowd_layer(z)
            z = z.transpose(0, 1).contiguous()
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward_in_train_step(x).view(-1, 10)
        y = y.transpose(1, 2).contiguous().view(-1).type(torch.int64)
        loss = F.cross_entropy(z, y, ignore_index=-1)
        return {"loss": loss, "log": {"train_loss": loss}}


class LabelSelectionModel(BasicModel):
    def __init__(
        self,
        num_embeddings: int,
        embeddings: np.ndarray,
        dropout_ratio: float = 0.5,
        n_labels: Optional[int] = None,
        n_workers: Optional[int] = None,
        learning_rate: float = 1e-3,
        embedding_dim: int = 300,
        max_sequence_length: int = 109,
        c: float = 0.8,
        lambda_: float = 32.0,
        selective_mode: str = "simple",
    ):
        if n_labels is None:
            raise ValueError("must set n_labels.")

        if n_workers is None:
            raise ValueError("must set n_workers.")

        super().__init__(
            num_embeddings=num_embeddings,
            embeddings=embeddings,
            dropout_ratio=dropout_ratio,
            n_labels=n_labels,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
        )
        self.save_hyperparameters(ignore=["embeddings"])

        if self.hparams["selective_mode"] == "simple":
            self.W = nn.Parameter(torch.Tensor(np.zeros(self.hparams["n_workers"])))

        elif self.hparams["selective_mode"] == "class-wise":
            self.W = nn.Parameter(
                torch.Tensor(
                    np.zeros((self.hparams["n_labels"], self.hparams["n_workers"]))
                )
            )

        elif self.hparams["selective_mode"] == "feature-based":
            self.selector = nn.Sequential(
                nn.Linear(50, self.hparams["n_workers"]),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

        self.selection_layer = nn.Sigmoid()

    def features(self, x):
        z = self.encode(x)
        z = z.transpose(2, 1)
        z = self.conv(z)
        z = z.transpose(2, 1)
        z = self.gru(z)[0]
        return z

    def model(self, x):
        z = self.features(x)
        return self.outputs(z)

    def forward(self, x):
        z = self.model(x)
        return z

    def selection_prob(self, x=None):
        if self.hparams["selective_mode"] in {"simple", "class-wise"}:
            return self.selection_layer(self.W)

        elif self.hparams["selective_mode"] == "feature-based":
            z = self.features(x)
            return self.selector(z)

        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        s = self.selection_prob(x)

        batch_size = y.shape[0]

        z_ = (
            z.repeat(self.hparams.n_workers, 1, 1, 1)
            .transpose(0, 1)
            .transpose(1, 2)
            .contiguous()
            .reshape(-1, 10)
        )

        if self.hparams["selective_mode"] == "simple":
            y_ = y.reshape(-1).type(torch.int64)
            s_ = s.repeat(batch_size, self.hparams.max_sequence_length, 1).view(-1)

        elif self.hparams["selective_mode"] == "class-wise":
            # NOTE: torch.gather の処理で input と index をいずれも 4 階のテンソルにするために dim=3 に次元を追加する。
            y = y.unsqueeze(3)
            y_ = y.reshape(-1).type(torch.int64)

            # inp_.shape: [batch_size, max_sequence_length, n_workers, n_labels]
            inp_ = s.repeat(
                batch_size, self.hparams["max_sequence_length"], 1, 1
            ).transpose(2, 3)

            # NOTE: y は -1 も取り得るが、-1 が入ると gather 部分で error となるので一旦 0 に置き換えている。
            # idx_.shape: [batch_size, max_sequence_length, n_workers, 1]
            idx_ = torch.where(y != -1, y, 0).type(torch.int64)

            s_ = torch.where(
                y != -1, torch.gather(input=inp_, dim=3, index=idx_), 0
            ).view(-1)

        elif self.hparams["selective_mode"] == "feature-based":
            y_ = y.reshape(-1).type(torch.int64)
            s_ = s.view(-1)

        else:
            raise ValueError

        ell = F.cross_entropy(z_, y_, ignore_index=-1, reduce=False)
        phi = s_[y_ != -1].mean()
        Am_size = (y_ != -1).sum()
        L = (ell.mul(s_).sum() / Am_size) / phi
        psi = torch.pow(torch.max(torch.tensor(0), self.hparams.c - phi), 2)
        loss = L + self.hparams["lambda_"] * psi
        return {"loss": loss, "log": {"train_loss": loss}}

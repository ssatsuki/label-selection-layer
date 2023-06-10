import numpy as np
import torch
from torch import nn


class CrowdLayer(nn.Module):
    '''The implementation of CrowdLayer proposed by "Deep Learning from Crowds"'''

    __constants__ = ["n_labels", "n_workers"]

    def __init__(self, n_labels: int, n_workers: int, mode: str = "MW"):
        super().__init__()
        self.n_labels = n_labels
        self.n_workers = n_workers
        self.mode = mode

        if self.mode == "MW":
            self.W = nn.Parameter(
                torch.Tensor(
                    np.tile(np.identity(self.n_labels), reps=(self.n_workers, 1, 1))
                )
            )

        elif self.mode == "VW":
            self.W = nn.Parameter(
                torch.Tensor(np.ones((self.n_workers, self.n_labels)))
            )

        elif self.mode == "VB":
            self.bias = nn.Parameter(
                torch.Tensor(np.zeros((self.n_workers, self.n_labels)))
            )

        elif self.mode == "VW+B":
            self.W = nn.Parameter(
                torch.Tensor(np.ones((self.n_workers, self.n_labels)))
            )
            self.bias = nn.Parameter(
                torch.Tensor(np.zeros((self.n_workers, self.n_labels)))
            )

        else:
            raise Exception("Unknown mode!")

    def forward(self, x):
        if self.mode == "MW":
            latent_features = torch.matmul(x, self.W)

        elif self.mode == "VW":
            latent_features = torch.stack(
                [torch.mul(self.W[i], x) for i in range(self.n_workers)]
            )

        elif self.mode == "VB":
            latent_features = torch.stack(
                [self.bias[i] + x for i in range(self.n_workers)]
            )

        elif self.mode == "VW+B":
            latent_features = torch.stack(
                [torch.mul(self.W[i], x) + self.bias[i] for i in range(self.n_workers)]
            )

        else:
            raise Exception("Unknown mode!")

        return latent_features

    def extra_repr(self) -> str:
        return f"n_labels={self.n_labels}, n_workers={self.n_workers}, mode={self.mode}"


class CrowdRegressionLayer(nn.Module):
    """The implementation of CrowdLayer, which proposed by "Deep Learning from Crowds," for Regression"""

    __constants__ = ["n_workers"]

    def __init__(self, n_workers: int, mode: str = "S"):
        super().__init__()
        self.n_workers = n_workers
        self.mode = mode

        if self.mode == "S":
            self.W = nn.Parameter(torch.Tensor(np.ones(self.n_workers)))

        elif self.mode == "B":
            self.bias = nn.Parameter(torch.Tensor(np.zeros(self.n_workers)))

        elif self.mode == "S+B":
            self.W = nn.Parameter(torch.Tensor(np.ones(self.n_workers)))
            self.bias = nn.Parameter(torch.Tensor(np.zeros(self.n_workers)))

        else:
            raise Exception("Unknown mode!")

    def forward(self, x):
        if self.mode == "S":
            latent_features = torch.stack(
                [torch.mul(self.W[i], x) for i in range(self.n_workers)]
            )

        elif self.mode == "B":
            latent_features = torch.stack(
                [self.bias[i] + x for i in range(self.n_workers)]
            )

        elif self.mode == "S+B":
            latent_features = torch.stack(
                [torch.mul(self.W[i], x) + self.bias[i] for i in range(self.n_workers)]
            )

        else:
            raise Exception("Unknown mode!")

        return latent_features

    def extra_repr(self) -> str:
        return f"n_workers={self.n_workers}, mode={self.mode}"

"""Helper classes for multi-dimensional edge feature-based AU relation graph (MEFARG) learning.

Implementation of building blocks in the MEFARG model from the paper:

    Luo, C., Song, S., Xie, W., Shen, L., Gunes, H. (2022). Learning multi-dimentionsal edge
    feature-based AU relation graph for facial action unit recognition. *arXiv*.
    `<https://arxiv.org/pdf/2205.01782.pdf>`_

Code adapted from the `OpenGraphAU <https://github.com/lingjivoo/OpenGraphAU/tree/main>`_ code base
(licensed under Apache 2.0).

"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class LinearBlock(nn.Module):
    """Apply transformations of multiple layers including a linear layer.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int, optional, default=None
        Size of each output sample. If `None`, uses `in_features`.
    drop: float, default=0.0
        Probability of dropping (zeroing out) input features.

    Notes
    -----
    Applies four transformations:

    - Linear
    - 1D batch normalization
    - ReLU
    - Drop out regularization

    Linear layer weights are initialized with :math:`N(0, \\sqrt{\\frac{2}{out\\_features}})`.
    Batch norm weights are initialized as 1 and biases as 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        # Layers
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        # Param init
        self.fc.weight.data.normal_(0, math.sqrt(2.0 / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


class AUPredictor(nn.Module):
    """Meta class for predicting action unit (AU) activations"""

    def __init__(
        self, in_features: int, n_main_nodes: int = 27, n_sub_nodes: int = 14
    ):
        super().__init__()
        self.in_features = in_features
        self.n_main_nodes = n_main_nodes
        self.n_sub_nodes = n_sub_nodes

        # Similarity calculation params
        self.main_sc = nn.Parameter(
            torch.FloatTensor(torch.zeros(self.n_main_nodes, self.in_features))
        )

        self.sub_sc = nn.Parameter(
            torch.FloatTensor(torch.zeros(self.n_sub_nodes, self.in_features))
        )
        self.relu = nn.ReLU()

        # List of AUs for which to generated sub AU (left, right) activations
        self.sub_list = (0, 1, 2, 4, 7, 8, 11)

        # Param init
        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    # pylint: disable=too-many-locals
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape

        # eq. 2
        main_sc = self.main_sc
        main_sc = self.relu(main_sc)
        main_sc = F.normalize(main_sc, p=2, dim=-1)
        main_cl = F.normalize(x, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []

        for i, index in enumerate(self.sub_list):
            # Calc sub node activations as left and right versions of main nodes
            au_l = 2 * i
            au_r = 2 * i + 1

            # eq. 2
            main_au = F.normalize(x[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            sub_sc = torch.stack([sc_l, sc_r], dim=0)

            cl = (main_au.view(1, b, c) * sub_sc.view(2, 1, c)).sum(dim=-1)

            sub_cl.append(cl[0].unsqueeze(1))
            sub_cl.append(cl[1].unsqueeze(1))

        sub_cl = torch.cat(sub_cl, dim=-1)

        return torch.cat([main_cl, sub_cl], dim=-1)

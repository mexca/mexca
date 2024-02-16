"""Multi-dimensional edge feature-based AU relation graph (MEFARG) learning.

Implementation of the MEFARG model from the paper:

    Luo, C., Song, S., Xie, W., Shen, L., Gunes, H. (2022). Learning multi-dimentionsal edge
    feature-based AU relation graph for facial action unit recognition. *arXiv*.
    `<https://arxiv.org/pdf/2205.01782.pdf>`_

Code adapted from the `OpenGraphAU <https://github.com/lingjivoo/OpenGraphAU/tree/main>`_ code base
(licensed under Apache 2.0).

"""

import logging
from typing import Dict

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from mexca.video.helper_classes import LinearBlock
from mexca.video.mefl import MEFL


class MEFARG(nn.Module, PyTorchModelHubMixin):
    """Apply a multi-dimensional edge feature-based action unit (AU) relation graph (MEFARG) model.

    Predict activations of 27 main and 14 sub AUs from representations of a face image.

    Parameters
    ----------
    config: dict
        Configuration dict of the model with two keys:
        `n_main_aus` is the number of main and `n_sub_aus` is the number of sub AUs to be predicted.
        If pretrained model weights are loaded, these must match the configuration.

    Notes
    -----
    First generates face representations using a ResNet50 backbone (default weights),
    then transforms them into AU activations using a multi-dimensional edge feature learning (MEFL) head (see :func:`mexca.video.mefl.mefl`).

    """

    def __init__(self, config: Dict):
        super().__init__()
        self.logger = logging.getLogger("mexca.video.MEFARG")
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove last FC layer to generate representations
        self.n_in_channels = 2048
        self.n_out_channels = self.n_in_channels // 4

        # Connect backbone with MEFL head
        self.linear_global = LinearBlock(
            self.n_in_channels, self.n_out_channels
        )
        self.head = MEFL(
            self.n_out_channels, config["n_main_aus"], config["n_sub_aus"]
        )

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform forward pass of backbone without last FC layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

    def forward(self, x):
        x = self._backbone_forward(x)

        b, c, _, _ = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)

        x = self.linear_global(x)
        cl = self.head(x)

        return cl

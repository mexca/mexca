"""Multi-dimensional edge feature-based AU relation graph (MEFARG) learning.

Implementation of the MEFARG model from the paper:

    Luo, C., Song, S., Xie, W., Shen, L., Gunes, H. (2022). Learning multi-dimentionsal edge
    feature-based AU relation graph for facial action unit recognition. *arXiv*.
    `<https://arxiv.org/pdf/2205.01782.pdf>`_

Code adapted from the `OpenGraphAU <https://github.com/lingjivoo/OpenGraphAU/tree/main>`_ code base
(licensed under Apache 2.0).

"""

import logging
import os
from collections import OrderedDict

import gdown
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from mexca.video.helper_classes import LinearBlock
from mexca.video.mefl import MEFL


class MEFARG(nn.Module):
    """Apply a multi-dimensional edge feature-based action unit (AU) relation graph (MEFARG) model.

    Predict activations of 27 main and 14 sub AUs from representations of a face image.

    Parameters
    ----------
    n_main_aus: int, default=27
        Number of main AUs.
    n_sub_aus: int, default=14
        Number of sub AUs.

    Notes
    -----
    First generates face representations using a ResNet50 backbone (default weights),
    then transforms them into AU activations using a multi-dimensional edge feature learning (MEFL) head (see :func:`mexca.video.mefl.mefl`).

    """

    def __init__(self, n_main_aus=27, n_sub_aus=14):
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
        self.head = MEFL(self.n_out_channels, n_main_aus, n_sub_aus)

    @classmethod
    def from_pretrained(cls, device: torch.device = torch.device(type="cpu")):
        """Load a pretrained model.

        If not found, the pretrained model is downloaded from Google Drive and stored in the PyTorch cache.

        Parameters
        ----------
        device: torch.device, default=torch.device(type='cpu')
            Device on which the model is loaded.

        """
        # Init class instance
        model = cls()

        model_id = "1UMnpbj_YKlqHF1m0DHV0KYD3qmcOmeXp"

        # Store pretraned model int PyTorch cache
        model_path = os.path.join(torch.hub.get_dir(), f"mefarg-{model_id}.pth")

        # Download model from Google Drive
        if not os.path.exists(model_path):
            url = f"https://drive.google.com/uc?&id={model_id}&confirm=t"
            model.logger.info("Downloading pretrained model from %s", url)
            gdown.download(url, output=model_path)
            model.logger.info("Pretrained model saved at %s", model_path)

        # Load pretrained state dict to device
        checkpoint = torch.load(model_path, map_location=device)

        # Replace state dict keys with correct model attribute names
        new_state_dict = OrderedDict()

        replace_dict = {
            "global_linear": "linear_global",
            "U": "linear_u",
            "V": "linear_v",
            "B": "linear_b",
            "E": "linear_e",
            "FAM": "fam",
            "ARM": "arm",
            "A": "linear_a",
            "head.main_class_linears": "head.main_node_linear_layers",
        }

        for state_k, state_v in checkpoint["state_dict"].items():
            if "module." in state_k:
                state_k = state_k[7:]  # Remove `module.` from keys

            for rep_k, rep_v in replace_dict.items():
                state_k = state_k.replace(rep_k, rep_v)

            new_state_dict[state_k] = state_v

        model.load_state_dict(new_state_dict, strict=True)

        return model

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

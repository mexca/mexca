"""Multi-dimensional edge feature learning (MEFL).

Implementation of the MEFL module from the paper:

    Luo, C., Song, S., Xie, W., Shen, L., Gunes, H. (2022). Learning multi-dimentionsal edge
    feature-based AU relation graph for facial action unit recognition. *arXiv*.
    `<https://arxiv.org/pdf/2205.01782.pdf>`_

Code adapted from the `OpenGraphAU <https://github.com/lingjivoo/OpenGraphAU/tree/main>`_ code base
(licensed under Apache 2.0).

"""

import logging
import math
import os
from collections import OrderedDict
from typing import Tuple
import gdown
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision.models import ResNet50_Weights, resnet50
from mexca.video.anfl import LinearBlock


class CrossAttention(nn.Module):
    """Apply a cross-attention layer.

    Parameters
    ----------
    in_features: int
        Size of each input sample.

    Notes
    -----
    Performs cross-attention between two inputs *x* and *y* as defined in eq. 4 of
    the corresponding `paper <https://arxiv.org/abs/2205.01782>`_.
    Linear layer weights are initialized with :math:`N(0, \\sqrt{\\frac{2}{out\\_features}})`.

    """
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        # Query layer
        self.linear_q = nn.Linear(in_features, in_features // 2)
        # Key layer
        self.linear_k = nn.Linear(in_features, in_features // 2)
        # Value layer
        self.linear_v = nn.Linear(in_features, in_features)
        self.scale = (self.in_features // 2) ** -0.5
        # Attention function
        self.attention = nn.Softmax(dim=-1)

        # Param init
        self.linear_k.weight.data.normal_(0, math.sqrt(2.0 / (in_features // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2.0 / (in_features // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2.0 / in_features))

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        # Key scoring
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attention(dots)
        # Attention weighting
        out = torch.matmul(attn, value)
        return out


class GraphEdgeModel(nn.Module):
    """Learn the relationships between nodes in a graph.

    Graph edge model: This class combines facial display-specific action unit representation modeling (FAM; i.e., cross-attention)
    with AU relationship modeling (ARM).

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    n_nodes: int
        Number of nodes in the graph.

    Notes
    -----
    Linear layer weights are initialized with :math:`N(0, \\sqrt{\\frac{2}{out\\_features}})`.
    Batch norm weights are initialized as 1 and biases as 0.

    """
    def __init__(self, in_features: int, n_nodes: int):
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes
        # Facial display-specific AU representation modelling
        self.fam = CrossAttention(self.in_features)
        # AU relationship modelling
        self.arm = CrossAttention(self.in_features)
        # Project edge features to AU relation graph
        self.edge_proj = nn.Linear(self.in_features, self.in_features)
        self.bn = nn.BatchNorm2d(self.n_nodes * self.n_nodes)

        # Param init
        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / self.in_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()


    def forward(self, node_feature: torch.Tensor, global_feature: torch.Tensor) -> torch.Tensor:
        # Global feature: Face representation from backbone
        b, n, d, c = node_feature.shape
        global_feature = global_feature.repeat(1, n, 1).view(b, n, d, c)
        # Transform node features
        feat = self.fam(node_feature, global_feature)
        feat_end = feat.repeat(1, 1, n, 1).view(b, -1, d, c)
        feat_start = feat.repeat(1, n, 1, 1).view(b, -1, d, c)
        # Calc node relationships
        feat = self.arm(feat_start, feat_end)
        # Project to AU graph
        edge = self.bn(self.edge_proj(feat))
        return edge


class GatedGNNLayer(nn.Module):
    """Apply a gated graph neural network (GNN) layer.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    n_nodes: int
        Number of nodes in the graph.
    dropout_rate: float, default=0.1
        Rate parameter of the dropout layer.

    Notes
    -----
    Performs gated graph convolution according to `Bresson and Laurent (2018, eq. 11) <https://arxiv.org/pdf/1711.07553.pdf>`_.
    Linear layer weights are initialized with :math:`N(0, \\sqrt{\\frac{2}{out\\_features}})`.
    Batch norm weights are initialized as 1 and biases as 0.

    """
    def __init__(self, in_features: int, n_nodes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes

        dim_in = self.in_features
        dim_out = self.in_features
        # GNN layers
        self.linear_u = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_out, bias=False)
        # Gating layers
        self.linear_a = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_b = nn.Linear(dim_in, dim_out, bias=False)
        # Edge layer
        self.linear_e = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.bnv = nn.BatchNorm1d(n_nodes)
        self.bne = nn.BatchNorm1d(n_nodes * n_nodes)

        self.act = nn.ReLU()
        # Param init
        self._init_weights_linear(dim_in)


    def _init_weights_linear(self, dim_in: int, gain: float = 1.0):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.linear_u.weight.data.normal_(0, scale)
        self.linear_v.weight.data.normal_(0, scale)
        self.linear_a.weight.data.normal_(0, scale)
        self.linear_b.weight.data.normal_(0, scale)
        self.linear_e.weight.data.normal_(0, scale)
    
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()
        self.bne.weight.data.fill_(1)
        self.bne.bias.data.zero_()


    def forward(self, x: torch.Tensor, edge: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep inputs
        res = x
        # Gating mechanism
        v_ix = self.linear_a(x)
        v_jx = self.linear_b(x)
        e = self.linear_e(edge)

        edge = edge + self.act(self.bne(torch.einsum('ev, bvc -> bec', (end, v_ix)) + torch.einsum('ev, bvc -> bec',(start, v_jx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.n_nodes, self.n_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        # GNN convolution mechanism
        u_jx = self.linear_v(x)  # V x H_out
        u_jx = torch.einsum('ev, bvc -> bec', (start, u_jx))  # E x H_out
        u_ix = self.linear_u(x)  # V x H_out
        x = u_ix + torch.einsum('ve, bec -> bvc', (end.t(), e * u_jx)) / self.n_nodes  # V x H_out
        x = res + self.act(self.bnv(x))

        return x, edge


class GatedGNN(nn.Module):
    """Apply multiple gated graph neural network (GNN) layers.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    n_nodes: int
        Number of nodes in the graph.
    n_layers: int, default=2
        Number of gated GNN layers.

    Notes
    -----
    Performs gated graph convolution according to Bresson and Laurent (2018, eq. 11) for multiple layers.

    """
    def __init__(self, in_features: int, n_nodes: int, n_layers: int = 2):
        super().__init__()
        self.in_features = in_features
        self.n_nodes = n_nodes
        # Init edge feature params
        start = torch.diagflat(torch.ones(self.n_nodes)).repeat(self.n_nodes, 1)
        end = torch.diagflat(torch.ones(self.n_nodes)).repeat_interleave(self.n_nodes, dim = 0)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)
        # Init gated GNN layers
        graph_layers = [GatedGNNLayer(self.in_features, self.n_nodes) for _ in range(n_layers)]

        self.graph_layers = nn.ModuleList(graph_layers)


    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = x.get_device()
        if dev >= 0:
            self.start = self.start.to(dev)
            self.end = self.end.to(dev)
        for _, layer in enumerate(self.graph_layers):
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge


class MEFL(nn.Module):
    """Apply multi-dimentional edge feature learning.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    n_main_nodes: int, default=27
        Number of main nodes in the facial graph.
    n_sub_nodes: int, default=14
        Number of sub nodes in the facial graph.

    Notes
    -----
    First learns node features via a series of linear layers for each main graph node.
    It then learns node relationships between graph nodes via graph edge modeling (GEM; cross-attention)
    from inputs and node features. Node features are transformed via global average pooling (GAP) and forwarded
    to multiple gated GNN layers together with the node relationship weights. Finally,
    a similarity calculation (SC) layer (cosine similarity) is applied to predict node activations.
    Sub node activations are calulated based on matching main node features.
    SC layer weights are initialized using Glorot initialization (see :func:`torch.nn.init.xavier_uniform`).

    """
    def __init__(self, in_features: int, n_main_nodes: int = 27, n_sub_nodes: int = 14):
        super().__init__()
        self.in_features = in_features
        self.n_main_nodes = n_main_nodes
        self.n_sub_nodes = n_sub_nodes

        # FC layers from AFG block
        self.main_node_linear_layers = nn.ModuleList(
            [
                LinearBlock(self.in_features, self.in_features)
                for _ in range(self.n_main_nodes)
            ]
        )

        self.edge_extractor = GraphEdgeModel(self.in_features, n_main_nodes)
        self.gnn = GatedGNN(self.in_features, n_main_nodes, 2)

        # Similarity calculation params
        self.main_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.n_main_nodes, self.in_features)))
        self.sub_sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.n_sub_nodes, self.in_features)))
        self.sub_list = [0,1,2,4,7,8,11]

        self.relu = nn.ReLU()
        # Init params
        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)


    def forward(self, x: torch.Tensor) -> torch.Tensor: # pylint: disable=too-many-locals
        # AFG mechanism
        f_u = [layer(x).unsqueeze(1) for layer in self.main_node_linear_layers]
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        # Edge feature mechanism
        f_e = self.edge_extractor(f_u, x)
        # Global average pooling
        f_e = f_e.mean(dim=-2)
        # Gated GNN mechanism
        f_v, f_e = self.gnn(f_v, f_e)

        b, n, c = f_v.shape

        # eq. 2
        main_sc = self.main_sc
        main_sc = self.relu(main_sc)
        main_sc = F.normalize(main_sc, p=2, dim=-1)
        main_cl = F.normalize(f_v, p=2, dim=-1)
        main_cl = (main_cl * main_sc.view(1, n, c)).sum(dim=-1)

        sub_cl = []

        for i, index in enumerate(self.sub_list):
            # Calc sub node activations as left and right versions of main nodes
            au_l = 2 * i
            au_r = 2 * i + 1

            # eq. 2
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)

            sc_l = F.normalize(self.relu(self.sub_sc[au_l]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[au_r]), p=2, dim=-1)

            sub_sc = torch.stack([sc_l, sc_r], dim=0)

            cl = (main_au.view(1, b, c) * sub_sc.view(2, 1, c)).sum(dim=-1)

            sub_cl.append(cl[0].unsqueeze(1))
            sub_cl.append(cl[1].unsqueeze(1))

        sub_cl = torch.cat(sub_cl, dim=-1)

        return torch.cat([main_cl, sub_cl], dim=-1)


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
    then transforms them into AU activations using a multi-dimensional edge feature learning (MEFL) head.

    """

    def __init__(self, n_main_aus=27, n_sub_aus=14):
        super().__init__()
        self.logger = logging.getLogger("mexca.video.MEFARG")
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() # Remove last FC layer to generate representations
        self.n_in_channels = 2048
        self.n_out_channels = self.n_in_channels // 4

        # Connect backbone with MEFL head
        self.linear_global = LinearBlock(self.n_in_channels, self.n_out_channels)
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
            # "head.gnn": "head.fgg.gnn",
            "U": "linear_u",
            "V": "linear_v",
            "B": "linear_b",
            "E": "linear_e",
            "FAM": "fam",
            "ARM": "arm",
            "A": "linear_a",
            # "head.main_sc": "head.fgg.main_sc",
            # "head.sub_sc": "head.fgg.sub_sc",
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
    
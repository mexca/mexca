"""Test video MEFL classes and methods.
"""

import pytest
import torch

from mexca.video.mefl import (
    MEFL,
    CrossAttention,
    GatedGNN,
    GatedGNNLayer,
    GraphEdgeModel,
)


class TestCrossAttention:
    in_features = 5
    n_nodes = 5
    n_dim = 3
    n_batch = 1

    @pytest.fixture
    def inputs_x(self):
        return torch.rand((self.n_batch, self.n_dim, self.in_features))

    @pytest.fixture
    def inputs_y(self):
        return torch.rand(
            (self.n_batch, self.n_nodes, self.n_dim, self.in_features)
        )

    @pytest.fixture
    def cross_attention(self):
        return CrossAttention(in_features=self.in_features)

    def test_forward(self, cross_attention, inputs_x, inputs_y):
        outputs = cross_attention.forward(inputs_x, inputs_y)
        assert outputs.shape == inputs_y.shape


class TestGraphEdgeModel(TestCrossAttention):
    @pytest.fixture
    def graph_edge_model(self):
        return GraphEdgeModel(self.in_features, self.n_nodes)

    def test_forward(self, graph_edge_model, inputs_x, inputs_y):
        outputs = graph_edge_model.forward(inputs_y, inputs_x)
        assert outputs.shape == (
            self.n_batch,
            self.n_nodes**2,
            self.n_dim,
            self.in_features,
        )


class TestGatedGNNLayer(TestGraphEdgeModel):
    @pytest.fixture
    def inputs_x(self):
        return torch.rand((self.n_batch, self.n_nodes, self.in_features))

    @pytest.fixture
    def inputs_edge(self):
        return torch.rand((self.n_batch, self.n_nodes**2, self.in_features))

    @pytest.fixture
    def inputs_start(self):
        return torch.rand((self.n_nodes**2, self.n_nodes))

    @pytest.fixture
    def gated_gnn_layer(self):
        return GatedGNNLayer(self.in_features, self.n_nodes)

    def test_forward(
        self, gated_gnn_layer, inputs_x, inputs_edge, inputs_start
    ):
        outputs_x, outputs_edge = gated_gnn_layer.forward(
            inputs_x, inputs_edge, inputs_start, inputs_start
        )
        assert outputs_x.shape == inputs_x.shape
        assert outputs_edge.shape == inputs_edge.shape


class TestGatedGNN(TestGatedGNNLayer):
    @pytest.fixture
    def gated_gnn(self):
        return GatedGNN(self.in_features, self.n_nodes)

    def test_forward(self, gated_gnn, inputs_x, inputs_edge):
        outputs_x, outputs_edge = gated_gnn.forward(inputs_x, inputs_edge)
        assert outputs_x.shape == inputs_x.shape
        assert outputs_edge.shape == inputs_edge.shape


class TestMEFL(TestCrossAttention):
    n_main_nodes = 27
    n_sub_nodes = 14

    @pytest.fixture
    def mefl(self):
        return MEFL(self.in_features)

    def test_forward(self, mefl, inputs_x):
        outputs = mefl.forward(inputs_x)
        assert outputs.shape == (
            self.n_batch,
            self.n_main_nodes + self.n_sub_nodes,
        )

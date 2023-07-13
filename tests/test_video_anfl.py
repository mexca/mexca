"""Test video ANFL classes and methods.
"""

import pytest
import torch

from mexca.video.anfl import ANFL, GNN, AUFeatureGenerator, FacialGraphGenerator


class TestGNN:
    in_features = 5
    n_nodes = 3
    n_neighbors = 1

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, self.n_nodes, self.in_features))

    @pytest.fixture
    def adj_mat(self):
        return torch.rand((1, self.n_nodes, self.n_nodes))

    @pytest.fixture
    def gnn(self):
        return GNN(self.in_features, self.n_nodes, self.n_neighbors)

    def test_calc_adj_mat(self, gnn, inputs):
        adj_mat = gnn._calc_adj_mat(inputs, self.n_neighbors)
        assert adj_mat.shape == (1, self.n_nodes, self.n_nodes)

    def test_normalize_digraph(self, gnn, adj_mat):
        norm_adj_mat = gnn._normalize_digraph(adj_mat)
        assert norm_adj_mat.shape == adj_mat.shape

    def test_forward(self, gnn, inputs):
        with torch.no_grad():
            outputs = gnn.forward(inputs)
        assert outputs.shape == inputs.shape


class TestAUFeatureGenerator:
    in_features = 5
    out_features = 3

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, 2, self.in_features))

    @pytest.fixture
    def generator(self):
        return AUFeatureGenerator(self.in_features, self.out_features)

    def test_forward(self, generator, inputs):
        with torch.no_grad():
            outputs = generator.forward(inputs)
        assert outputs.shape == (1, self.out_features, self.in_features)


class TestFacialGraphGenerator:
    in_features = 5
    n_main_nodes = 27
    n_sub_nodes = 14
    n_neighbors = 1

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, self.n_main_nodes, self.in_features))

    @pytest.fixture
    def generator(self):
        return FacialGraphGenerator(
            self.in_features,
            self.n_main_nodes,
            self.n_sub_nodes,
            self.n_neighbors,
        )

    def test_forward(self, generator, inputs):
        with torch.no_grad():
            outputs = generator.forward(inputs)
        assert outputs.shape == (1, self.n_main_nodes + self.n_sub_nodes)


class TestANFL:
    in_features = 5
    n_main_nodes = 27
    n_sub_nodes = 14
    n_neighbors = 1

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, 2, self.in_features))

    @pytest.fixture
    def anfl(self):
        return ANFL(
            self.in_features,
            self.n_main_nodes,
            self.n_sub_nodes,
            self.n_neighbors,
        )

    def test_forward(self, anfl, inputs):
        with torch.no_grad():
            outputs = anfl.forward(inputs)
        assert outputs.shape == (1, self.n_main_nodes + self.n_sub_nodes)

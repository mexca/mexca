"""Test video helper classes and methods.
"""

import pytest
import torch

from mexca.video.helper_classes import AUPredictor, LinearBlock


class TestLinearBlock:
    in_features = 5
    out_features = 5

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, self.in_features, self.in_features))

    @pytest.fixture
    def linear_block_in_eq_out(self):
        return LinearBlock(in_features=self.in_features)

    @pytest.fixture
    def linear_block(self):
        return LinearBlock(
            in_features=self.in_features, out_features=self.out_features
        )

    def test_forward_in_eq_out(self, linear_block_in_eq_out, inputs):
        outputs = linear_block_in_eq_out.forward(inputs)
        assert outputs.shape == inputs.shape

    def test_forward(self, linear_block, inputs):
        outputs = linear_block.forward(inputs)
        assert outputs.shape == inputs.shape


class TestAUPredictor:
    in_features = 5
    n_main_nodes = 27
    n_sub_nodes = 14
    n_batch = 1

    @pytest.fixture
    def inputs(self):
        return torch.rand((self.n_batch, self.n_main_nodes, self.in_features))

    @pytest.fixture
    def au_predictor(self):
        return AUPredictor(
            self.in_features, self.n_main_nodes, self.n_sub_nodes
        )

    def test_forward(self, au_predictor, inputs):
        outputs = au_predictor.forward(inputs)
        assert outputs.shape == (
            self.n_batch,
            self.n_main_nodes + self.n_sub_nodes,
        )

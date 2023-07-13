"""Test video MEFARG classes and methods.
"""

import pytest
import torch

from mexca.video.mefarg import MEFARG


class TestMEFARG:
    in_features = 256
    n_main_nodes = 27
    n_sub_nodes = 14
    n_batch = 1

    @pytest.fixture
    def inputs(self):
        return torch.rand((1, 3, self.in_features, self.in_features))

    @pytest.fixture
    def mefarg(self):
        return MEFARG(self.n_main_nodes, self.n_sub_nodes)

    def test_from_pretrained(self):
        model = MEFARG.from_pretrained()
        assert isinstance(model, MEFARG)

    def test_forward(self, mefarg, inputs):
        with torch.no_grad():
            outputs = mefarg.forward(inputs)
        assert outputs.shape == (
            self.n_batch,
            self.n_main_nodes + self.n_sub_nodes,
        )

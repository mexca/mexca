""" Test utility functions """

import pytest
import numpy as np
from mexca.core.exceptions import TimeStepWarning
from mexca.core.utils import create_time_var_from_step


def test_create_time_var_from_step():
    time_var_float = create_time_var_from_step(0.04, 0.2)
    assert all(time_var_float == np.array([0.00, 0.04, 0.08, 0.12, 0.16]))

    time_var_int = create_time_var_from_step(1, 5)
    assert all(time_var_int == np.array([0, 1, 2, 3, 4]))

    with pytest.warns(TimeStepWarning):
        time_var_warn = create_time_var_from_step(0.04, 0.21)

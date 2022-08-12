"""Utility functions.
"""

from decimal import Decimal
import numpy as np
from mexca.core.exceptions import TimeStepWarning


def create_time_var_from_step(time_step, end_time):
    """Create a time variable from a step and duration.

    This function creates a time variable if none is available from video processing.

    Parameters
    ----------
    time_step: float
        The interval between time points.
    end_time: float
        The maximum time of the time variable. Should match the length of the video/audio file.

    Returns
    -------
    numpy.ndarray
        An array with time points.

    """
    not_processed = Decimal(end_time)%Decimal(time_step) # Use 'Decimal' to avoid issues with float representation

    if not_processed > 0.0:
        TimeStepWarning(f'Length of file is not a multiple of "time_step": {not_processed}s at the end of the file will not be processed')

    time = np.arange(start=0.0, stop=end_time, step=time_step)

    return time

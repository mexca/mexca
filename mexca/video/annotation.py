""" Classes and methods for video annotation """

import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class Face:
    frame: int
    time: float
    array: np.float32
    prob: float
    embeddings: np.float32
    label: str

    def __post_init__(self) -> None:
        self.check_attrs()


    def check_attrs(self) -> None:
        if isinstance(self.frame, int) and self.frame < 0:
            raise ValueError('Attribute "frame" cannot be lower than 0')
        if isinstance(self.time, float) and self.time < 0.0:
            raise ValueError('Attribute "time" cannote be lower than 0.0')


    def convert_to_list(self, attr) -> 'Any': # Only for testing
        if not isinstance(attr, (list, dict, str, int, float)) and attr is not None:
            return attr.tolist()

        return attr


    def to_dict(self) -> dict: # Only for testing
        new_dict = {}
        for key, value in asdict(self).items():
            if key not in ['array', 'embeddings']:
                new_dict[key] = self.convert_to_list(value)
        return new_dict

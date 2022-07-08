""" Output classes and methods """

class Multimodal:
    def __init__(self) -> 'Multimodal':
        self.features = {}


    def add(self, feature_dict, replace=False):
        if feature_dict:
            for key, val in feature_dict.items():
                if key not in self.features or replace:
                    self.features[key] = val

""" Output classes and methods """

import numpy as np
from scipy.optimize import linear_sum_assignment

class Multimodal:
    def __init__(self) -> 'Multimodal':
        self.features = {}


    def add(self, feature_dict, replace=False):
        if feature_dict:
            for key, val in feature_dict.items():
                if key not in self.features or replace:
                    self.features[key] = val


    def match_faces_speakers(self, face_label='label', speaker_label='speaker', id_label='id'):
        time = self.features['time']
        spks = list(set(self.features[speaker_label]))
        faces = list(set(self.features[face_label]))

        mat = np.zeros((len(spks), len(faces)))

        for i, spk in enumerate(spks):
            for j, face in enumerate(faces):
                matches = np.logical_and(
                    np.equal(
                        np.array(self.features[speaker_label], dtype=np.chararray), spk
                    ), np.equal(np.array(self.features[face_label]), face))
                mat[i, j] = np.sum(np.array(time)[matches])

        max_mapping = linear_sum_assignment(mat, maximize=True)

        face_speaker_id = np.zeros(len(time)) # Zero indicates no match

        for i, match in enumerate(np.array(max_mapping).T):
            face_speaker_id[np.logical_and(
                np.equal(np.array(self.features[speaker_label], dtype=np.chararray), spks[match[0]]),
                np.equal(np.array(self.features[face_label]), faces[match[1]])
            )] = i + 1

        self.features[id_label] = face_speaker_id

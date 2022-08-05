"""Store the pipeline output.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class Multimodal:
    """Store the pipeline output.
    """
    def __init__(self) -> 'Multimodal':
        """Create a class instance to store the pipeline output.

        Returns
        -------
        A ``Multimodal`` class instance.

        """
        self.features = {}


    def add(self, feature_dict, replace=False):
        """Add features to the pipeline output.

        Parameters
        ----------
        feature_dict: dict
            A dictionary with feature names as keys and feature values as values.
        replace: bool, default=False
            Whether existing features with the same names as in `feature_dict` should be replaced.

        """
        if feature_dict:
            for key, val in feature_dict.items():
                if key not in self.features or replace:
                    self.features[key] = val


    def match_faces_speakers(
            self,
            face_label='face_id',
            speaker_label='speaker_id',
            id_label='match_id'
        ): # pylint: disable=too-many-locals
        """Match face and speaker labels by time overlap.

        Performs a linear sum assignment using ``scipy.optimize.linear_sum_assignment`` by tallying
        the time overlap between faces and speakers. Matches face and speaker labels by maximum time overlap.
        Modifies the `Multimodal.feature` attribute in place.

        Parameters
        ----------
        face_label: str, default='face_id'
            The feature name of the face labels.
        speaker_label: str, default='speaker_id'
            The feature name of the speaker labels.
        id_label: str, default='match_id'
            The feature name of the matched labels.

        """
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

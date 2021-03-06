""" Audio speaker id and voice feature integration classes and methods """

import numpy as np


class AudioIntegrator:
    def __init__(self, identifier, extractor) -> 'AudioIntegrator':
        self.identifier = identifier
        self.extractor = extractor


    def integrate(self, audio_features, annotation):
        time = audio_features['time']

        annotated_features = audio_features
        annotated_features['segment_id'] = np.zeros_like(time)
        annotated_features['segment_start'] = np.zeros_like(time)
        annotated_features['segment_end'] = np.zeros_like(time)
        annotated_features['track'] = np.full_like(time, fill_value='', dtype=np.chararray)
        annotated_features['speaker_id'] = np.full_like(time, fill_value='', dtype=np.chararray)

        seg_idx = 1

        for seg, track, spk in annotation.itertracks(yield_label=True):
            is_segment = np.logical_and(
                np.less(time, seg.end), np.greater(time, seg.start)
            )
            annotated_features['segment_id'][is_segment] = seg_idx
            annotated_features['segment_start'][is_segment] = seg.start
            annotated_features['segment_end'][is_segment] = seg.end
            annotated_features['track'][is_segment] = track
            annotated_features['speaker_id'][is_segment] = spk

            seg_idx += 1

        return annotated_features


    def apply(self, filepath, time):
        annotation = self.identifier.apply(filepath)
        voice_features = self.extractor.extract_features(filepath, time)
        annotated_features = self.integrate(voice_features, annotation)

        return annotated_features

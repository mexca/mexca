"""Integrate output about speech segments, speakers, and voice features.
"""

import numpy as np
from tqdm import tqdm


class AudioIntegrator:
    """Integrate output about speech segments, speakers, and voice features.
    """
    def __init__(self, identifier, extractor) -> 'AudioIntegrator':
        """Create a class instance to integrate output about speech segments, speakers, and voice features.

        Parameters
        ----------
        identifier: mexca.audio.SpeakerIdentifier
            An instance of the ``mexca.audio.SpeakerIdentifier`` class.
        extractor: mexca.audio.VoiceExtractor
            An instance of the ``mexca.audio.VoiceExtractor`` class.

        Returns
        -------
        An ``AudioIntegrator`` class instance.

        """
        self.identifier = identifier
        self.extractor = extractor


    def integrate(self, audio_features, annotation, show_progress=True):
        """Combine extracted voice features with speech and speaker annotations.

        Parameters
        ----------
        audio_features: dict
            A dictionary with the extracted voice features.
        annotation: pyannote.core.Annotation
            A pyannote annotation object containing detected speech segments and speakers.

        Returns
        -------
        dict
            A dictionary with key-value pairs:
            - ``segment_id``: An int array indexing detected speech segments.
            - ``segment_start``: A float array with start times of detected speech segments.
            - ``segment_end``: A float array with end times of detected speech segments.
            - ``track`: A str array with labels of detected speech tracks.
            - ``speaker_id``: A str array with labels of detected speakers.

        """
        time = audio_features['time']

        annotated_features = audio_features
        annotated_features['segment_id'] = np.zeros_like(time)
        annotated_features['segment_start'] = np.zeros_like(time)
        annotated_features['segment_end'] = np.zeros_like(time)
        annotated_features['track'] = np.full_like(time, fill_value='', dtype=np.chararray)
        annotated_features['speaker_id'] = np.full_like(time, fill_value='', dtype=np.chararray)

        seg_idx = 1

        for seg, track, spk in tqdm(annotation.itertracks(yield_label=True), disable=not show_progress):
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


    def apply(self, filepath, time, show_progress=True):
        """Apply speech and speaker identification, voice feature extraction, and integration after each other.

        Parameters
        ----------
        filepath: str or path
            Path to the audio file.
        time: List or numpy.ndarray
            List or array with time points for with voice features should be extracted.
        show_progress bool, default=True:
            Enables a progress bar.

        Returns
        -------
        dict
            A dictionary with annotated voice features. See ``integrate`` method for details.

        """
        annotation = self.identifier.apply(filepath, show_progress)
        voice_features = self.extractor.extract_features(filepath, time)
        annotated_features = self.integrate(voice_features, annotation, show_progress)

        return annotated_features

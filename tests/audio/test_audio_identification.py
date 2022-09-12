""" Test speaker identification classes and methods """

import itertools
import json
import os
import pytest
from pyannote.core import Annotation
from mexca.audio.identification import SpeakerIdentifier


class TestSpeakerIdentifier:
    speaker_identifier = SpeakerIdentifier()
    filepath = os.path.join(
        'tests', 'test_files', 'test_eng_5_seconds.wav'
    )
    with open(os.path.join(
            'tests', 'reference_files', 'annotation_eng_5_seconds.json'
        ), 'r', encoding="utf-8") as file:
        ref_speakers = Annotation.from_json(json.loads(file.read()))


    def test_properties(self):
        with pytest.raises(TypeError):
            self.speaker_identifier.pyannote_audio = 3.0

        with pytest.raises(ValueError):
            self.speaker_identifier.num_speakers = -1

        with pytest.raises(TypeError):
            self.speaker_identifier.num_speakers = 'k'


    def test_apply(self):
        speakers = self.speaker_identifier.apply(self.filepath)
        track_pairs = itertools.zip_longest(
            speakers.itertracks(yield_label=True),
            self.ref_speakers.itertracks(yield_label=True)
        )
        for track, ref_track in track_pairs:
            assert pytest.approx(track[0].start, rel=1e-2) == ref_track[0].start
            assert pytest.approx(track[0].end, rel=1e-2) == ref_track[0].end
            assert track[1] == ref_track[1]
            assert track[2] == ref_track[2]

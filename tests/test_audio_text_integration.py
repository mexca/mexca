""" Test Audio text integration classes and methods """

import os
import json
from mexca.text.transcription import AudioTextIntegrator


class TestAudioTextIntegration:
    audio_text_integrator = AudioTextIntegrator(language='dutch')
    audio_filepath = os.path.join('tests', 'audio_files', 'test_dutch_5_seconds.wav')

    # reference output
    with open(os.path.join(
            'tests', 'reference_files', 'text_audio_integration.json'), 'r') as file:
        text_audio_transcription = json.loads(file.read())

    def test_apply(self):

        out  = self.audio_text_integrator.apply(self.audio_filepath)

        assert are_equal(out, self.text_audio_transcription) # test the output against the reference



def are_equal(reference_output, predicted_output):

    dict1_len = len(reference_output)
    dict2_len = len(predicted_output)
    total_dict_count = dict1_len + dict2_len
    shared_dict = {}

    equal = False
    for i in reference_output:
        if (i in predicted_output) and (reference_output[i] == predicted_output[i]):
            shared_dict[i] = reference_output[i]
    len_shared_dict=len(shared_dict)

    if (len_shared_dict == total_dict_count/2):
        equal = True
    return equal

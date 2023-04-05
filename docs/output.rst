Output
======

This section provides a detailed description of the emotion expression features that can be extracted with mexca in addition to other output.
The extracted features are returned as a 2-dimensional table with columns as features and rows as samples.
The merged features from all modalities can be accessed via the `features` attribute of the `Multimodal` class, which is returned by the pipeline.


General
-------

- `frame`: The index of the video frame for which features were extracted (starting at zero).
- `time`: The time stamp of the video frame for which features were extracted (in seconds).

Facial
------

Output extracted from faces displayed in the video has the prefix `face_`. A frame can show multiple faces for which features are extracted.
In this case, each face is shown in a separate row with the same `frame` and `time` values. 

- `face_box`: The bounding box for a single detected face. The box has four coordinates (x1, y1, x2, y2).
- `face_prob`: The probability with which the face was detected. 
- `face_landmarks`: The landmark coordinates for the detected face. The array contains 68 coordinate pairs (x, y).
- `face_aus`: The action unit (AU) activations for the detected face. The output differs between AU detection models: `svm` returns binary unit activations, whereas `xgb` returns continuous activations (from a tree ensemble) for 20 action units.
- `face_label`: The ID label of the detected face returned by the clustering of the face embeddings (starting at zero).
- `face_confidence`: A confidence score of the `face_label` assignment between 0 and 1: The normalized distance from each face embedding vector to it's cluster centroid relative to the distance to the nearest other cluster centroid.


.. _voice_features_output:

Voice
-----

Output of the audio processing contains voice features and information about the speaker.
A frame can have overlapping speakers for which features are extracted separately and added as separate rows.

- `segment_start`: The starting time stamp of the speech segment (in seconds).
- `segment_end`: The ending time stamp of the speech segment (in seconds).
- `segment_speaker_label`: The ID of the speaker returned by the clustering of the speaker embeddings (unique integer numbers).

By default, the following voice features are extracted:

- `pitch_f0_hz`: The voice pitch measured as the fundamental frequency F0 in Hz. Calculated using the probabilistic YIN method described in Mauch and Dixon (2014) as well as De Cheveigne and Kawahara (2002).
- `jitter_local_rel_F0`: The voice jitter measured as the average difference between consecutive pitch periods (i.e., time difference between glottal pulses) relative to the average pitch period (i.e., 1/F0).
- `shimmer_local_rel_f0`: The voice shimmer measured as the average difference between amplitudes of consecutive pitch periods relative to the average amplitude of the pitch periods.
- `hnr_db`: The harmonics-to-noise ratio in dB calculated using the autocorrelation function as described in Boersma (1993).
- `f1_freq_hz`: The central frequency of the first formant in Hz. Calculated using Burg's method by finding the roots of the linear predictive coefficients.
- `f1_bandwidth_hz`: The bandwidth of the first formant in Hz.
- `f1_amplitude_rel_f0`: The amplitude of the first formant relative to the amplitude of the closest F0 harmonic.

Details and further references on the feature extraction can be found in the description of the `GeMAPS <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7160715>`_ feature set (Eyben et al., 2016).

Text
----

Output extracted from the transcribed text has the prefix `span_`. By default, the text is split into sentences (i.e, *spans*),
and sentiment scores are predicted for each sentence.

- `span_start`: The time stamp where the token starts (in seconds).
- `span_end`: The time stamp where the token ends (in seconds).
- `span_text`: The text of the sentence (span).
- `span_sent_pos`: The positive sentiment score of the sentence (span).
- `span_sent_neg`: The negative sentiment score of the sentence (span).
- `span_sent_neu`: The neutral sentiment score of the sentence (span).

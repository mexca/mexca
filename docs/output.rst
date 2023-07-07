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
- `face_landmarks`: The landmark coordinates for the detected face. The array contains 5 coordinate pairs (x, y).
- `face_aus`: The action unit (AU) activations for the detected face. Contains 41 activations between 0 and 1. The first 27 values correspond to bilateral AUs: 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 32, 38, 39. The following 14 values correspond to unilateral AUs: L1, R1, L2, R2, L4, R4, L6, R6, L10, R10, L12, R12, L14, R14 (L = left, R = right sided activation). Unilateral are activations are computed based on bilateral activations of the same AU. **We recommend to only use AUs for which the prediction preformance has been validated** (see `Luo et al., 2022 <https://arxiv.org/pdf/2205.01782.pdf>`_).
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
- `f2_freq_hz`: The central frequency of the first formant in Hz.
- `f2_bandwidth_hz`: The bandwidth of the second formant in Hz.
- `f2_amplitude_rel_f0`: The amplitude of the second formant relative to the amplitude of the closest F0 harmonic.
- `f3_freq_hz`: The central frequency of the third formant in Hz.
- `f3_bandwidth_hz`: The bandwidth of the third formant in Hz.
- `f3_amplitude_rel_f0`: The amplitude of the third formant relative to the amplitude of the closest F0 harmonic.
- `alpha_ratio_db`: Alpha ratio in dB: Ratio of the summed energy in the frequency band `[50, 1000)` and the band `[1000, 5000)` (Hz).
- `hammar_index_db`: Hammarberg index in dB: Ratio of the peak magnitude in the frequency band `[0, 2000)` and the band `[2000, 5000)` (Hz).
- `spectral_slope_0_500`: Spectral slope in the frequency band `[0, 500)` Hz.
- `spectral_slope_500_1500`: Spectral slope in the frequency band `[500, 1500)` Hz.
- `h1_h2_diff_db`: Difference between the first and second pitch F0 harmonic amplitudes in dB.
- `h1_f3_diff_db`: Difference between the first and F0 harmonic and the third formant amplitudes in dB.
- `mfcc_1`: First Mel frequency cepstral coeffcient (MFCC).
- `mfcc_2`: Second MFCC.
- `mfcc_3`: Third MFCC.
- `mfcc_4`: Fourth MFCC.
- `spectral_flux`: Spectral flux: Sum of the squared first-order magnitude per-frequency-bin difference.
- `rms_db`: Root mean squared energy in dB.

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

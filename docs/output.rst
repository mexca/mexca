Output
======

This section provides a detailed description of the emotion expression features that can be extracted with **mexca** in addition to other output. The extracted features are returned as a 2-dimensional table with columns indicating the features.

General
-------

- `frame`: The index of the video frame for which features were extracted (starting at zero).
- `time`: The time stamp of the video frame for which features were extracted (in seconds).

Facial
------

Output extracted from faces displayed in the video has the prefix `face_`. A frame can show multiple faces for which features are extracted. In this case, each face is shown in a separate row with the same `frame` and `time` values. 

- `face_box`: The bounding box for a single detected face. The box has four coordinates (x1, y1, x2, y2).
- `face_prob`: The probability with which the face was detected. 
- `face_landmarks`: The landmark coordinates for the detected face. The array contains 68 coordinate pairs (x, y).
- `face_aus`: The action unit (AU) activations for the detected face. The output differs between AU detection models: `JAANET` returns intensities (0-1) for 12 action units, whereas `svm` and `logistic` return presence/absence (1/0) values for 20 action units.
- `face_id`: The ID of the detected face returned by the clustering of the face embeddings (starting at zero).

Voice
-----

Output of the audio processing contains voice features (currently only `pitchF0`) and information about the speaker. A frame can have overlapping speakers for which features are extracted separately and added as separate rows.

- `pitchF0`: The voice pitch measured as the fundamental frequency F0.
- `segment_id`: The ID of the speech segment returned by speaker segmentation model.
- `segment_start`: The starting time stamp of the speech segment (in seconds).
- `segment_end`: The ending time stamp of the speech segment (in seconds).
- `track`: The label of the speaker track (only useful when speakers are overlapping).
- `speaker_id`: The ID of the speaker returned by the clustering of the speaker embeddings (unique numbers).

Text
----

Output extracted from the spoken text processing has the prefix `text_`. The text is extracted as tokens (i.e, words) which usually change much slower than video and audio features. Therefore, tokens are repeated for rows where `time` matches their start and end to align the text with the other two modalities.

- `text_token_id`: The index of the text token (i.e., word, starting at zero).
- `text_token`: The text string of the token.
- `text_token_start`: The time stamp where the token starts (in seconds).
- `text_token_end`: The time stamp where the token ends (in seconds).
- `text_sent_id`: The index of the sentence the token belongs to.
- `text_sent_pos`: The positive sentiment score of the sentence the token belongs to.
- `text_sent_neg`: The negative sentiment score of the sentence the token belongs to.
- `text_sent_neu`: The neutral sentiment score of the sentence the token belongs to.

Other
-----

- `match_id`: The ID of the match between `face_id` and `speaker_id` based on the maximum overlapping time. A zero indicates that there is no match.

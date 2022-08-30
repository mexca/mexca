Features
========

The current release of **mexca** allows users to (i) identify faces and speakers (ii) extract facial expressions and voice features, and (iii) transcribe speech from video files. The package contains three submodules that process video, audio, and text.

Video
-----

The video processing includes four steps:

1. Detection: Faces displayed in the video frames are detected using a pretrained `MTCNN` model from `facenet-pytorch`.
2. Encoding: Faces are extracted from the frames and embeddings are computed using `InceptionResnetV1` from `facenet-pytorch`.
3. Identification: IDs are assigned to faces by clustering the embeddings using spectral clustering.
4. Extraction: Facial features (landmarks, action units) are extracted from the faces using `pyfeat`. Available models are `PFLD`, `MobileFaceNet`, and `MobileNet` for landmark extraction and `JAANET`, `svm`, and `logistic` for action unit extraction.

Audio
-----

The audio processing contains four steps, which are performed by the speaker diarization pipeline from `pyannote.audio`:

1. Segmentation: Speech segments from different speakers are detected using `pyannote/segmentation`.
2. Encoding: Speaker embeddings are computed for each speech segment using `ECAPA-TDNN` from `speechbrain`.
3. Identification: IDs are assigned to speech segments based on classification with a Gaussian hidden Markov model.
4. Extraction: Voice features are extracted using `praat-parselmouth`. Currently, only the voice pitch (F0) can be extracted.

Text
----

The speech text is trascribed from the audio signal using a fine-tuned `wave2vec` model from `huggingsound`. Languages that can be transcribed are English and Dutch.

References
----------

Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *arXiv*. https://doi.org/10.48550/arXiv.2104.04045

Cheong, J. H., Xie, T., Byrne, S., & Chang, L. J. (2021). Py-feat: Python facial expression analysis toolbox. *arXiv*. https://doi.org/10.48550/arXiv.2104.03509

Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., … Bengio, Y. (2021). SpeechBrain: A General-Purpose Speech Toolkit. *arXiv*. https://doi.org/10.48550/arXiv.2106.04624

Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). wav2vec: Unsupervised pre-training for speech recognition. *arXiv*. https://doi.org/10.48550/arXiv.1904.05862

Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *arXiv*. https://doi.org/10.48550/arXiv.1503.03832

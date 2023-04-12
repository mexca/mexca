Components
==========

The mexca package contains five components that can be used to build the MEXCA pipeline.


FaceExtractor
-------------

This component takes a video file as input as and applies four steps:

1. Detection: Faces displayed in the video frames are detected using a pretrained `MTCNN` model from `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ [#]_.
2. Encoding: Faces are extracted from the frames and encoded into an embedding space using `InceptionResnetV1` from `facenet-pytorch`.
3. Identification: IDs are assigned to faces by clustering the embeddings using spectral clustering (k-means).
4. Extraction: Facial features (landmarks, action units) are extracted from the faces using `pyfeat <https://py-feat.org/pages/intro.html>`_ [#]_. Available models are `PFLD`, `MobileFaceNet`, and `MobileNet` for landmark extraction and `svm`, and `xgb` for action unit extraction.

.. note::
    The two available AU extraction models give different output: `svm` returns binary unit activations, whereas `xgb` returns continuous activations (from a tree ensemble).


SpeakerIdentifier
-----------------

This component takes an audio file as input and applies three steps using the speaker diarization pipeline from `pyannote.audio <https://github.com/pyannote/pyannote-audio>`_ [#]_:

1. Segmentation: Speech segments are detected using `pyannote/segmentation` (this step includes voice activity detection).
2. Encoding: Speaker embeddings are computed for each speech segment using `ECAPA-TDNN` from `speechbrain <https://speechbrain.github.io/#>`_ [#]_.
3. Identification: IDs are assigned to speech segments based on clustering with a Gaussian hidden Markov model.


VoiceExtractor
--------------

This component takes the audio file as input and extracts voice features using `librosa <https://librosa.org/doc/latest/index.html>`_ [#]_. 
For the default set of voice features that are extracted, see the :ref:`output <voice_features_output>` section.


AudioTranscriber
----------------

This component takes the audio file and speech segments information as input.
It transcribes the speech segments to text using a pretrained `Whisper <https://github.com/openai/whisper>`_ model.
The resulting transcriptions are aligned with the speaker segments. The transcriptions are split into sentences using a regular expression.

SentimentExtractor
------------------

This component takes the transcribed text sentences as input and predicts sentiment scores (positive, negative, neutral) for each sentence
using a pretrained multilingual RoBERTa `model <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment>`_ [#]_.

References
----------

.. [#] Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *arxiv*. https://doi.org/10.48550/arxiv.2010.12421

.. [#] Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *arXiv*. https://doi.org/10.48550/arXiv.2104.04045

.. [#] Cheong, J. H., Xie, T., Byrne, S., & Chang, L. J. (2021). Py-feat: Python facial expression analysis toolbox. *arXiv*. https://doi.org/10.48550/arXiv.2104.03509

.. [#] McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In *Proceedings of the 14th Python in Science Conference*, 18-25.

.. [#] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. https://cdn.openai.com/papers/whisper.pdf

.. [#] Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., … Bengio, Y. (2021). SpeechBrain: A general-purpose speech toolkit. *arXiv*. https://doi.org/10.48550/arXiv.2106.04624

.. [#] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *arXiv*. https://doi.org/10.48550/arXiv.1503.03832

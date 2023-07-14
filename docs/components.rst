Components
==========

The mexca package contains five components that can be used to build the MEXCA pipeline.


FaceExtractor
-------------

This component takes a video file as input as and applies four steps:

1. Detection: Faces displayed in the video frames are detected using a pretrained `MTCNN` model from `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ [1]_.
2. Encoding: Faces are extracted from the frames and encoded into an embedding space using `InceptionResnetV1` from `facenet-pytorch`.
3. Identification: IDs are assigned to faces by clustering the embeddings using spectral clustering (k-means).
4. Extraction: Facial landmarks are extracted using the pretrained MTCNN from *facenet-pytorch*. Facial action unit activations are extracted using a pretrained Multi-dimensional Edge Feature-based AU Relation Graph model which is adpated from the `OpenGraphAU <https://github.com/lingjivoo/OpenGraphAU>`_ code base [2]_. Currently, only the ResNet-50 backbone is available.


SpeakerIdentifier
-----------------

This component takes an audio file as input and applies three steps using the speaker diarization pipeline from `pyannote.audio <https://github.com/pyannote/pyannote-audio>`_ [3]_:

1. Segmentation: Speech segments are detected using `pyannote/segmentation` (this step includes voice activity detection).
2. Encoding: Speaker embeddings are computed for each speech segment using `ECAPA-TDNN` from `speechbrain <https://speechbrain.github.io/#>`_ [4]_.
3. Identification: IDs are assigned to speech segments based on clustering with a Gaussian hidden Markov model.


VoiceExtractor
--------------

This component takes the audio file as input and extracts voice features using `librosa <https://librosa.org/doc/latest/index.html>`_ [5]_.
For the default set of voice features that are extracted, see the :ref:`output <voice_features_output>` section.


AudioTranscriber
----------------

This component takes the audio file and speech segments information as input.
It transcribes the speech segments to text using a pretrained `Whisper <https://github.com/openai/whisper>`_ model [6]_.
The resulting transcriptions are aligned with the speaker segments. The transcriptions are split into sentences using a regular expression.


SentimentExtractor
------------------

This component takes the transcribed text sentences as input and predicts sentiment scores (positive, negative, neutral) for each sentence
using a pretrained multilingual RoBERTa `model <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment>`_ [7]_.

References
----------

.. [7] Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *arxiv*. https://doi.org/10.48550/arxiv.2010.12421

.. [3] Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *arXiv*. https://doi.org/10.48550/arXiv.2104.04045

.. [2] Luo, C., Song, S., Xie, W., Shen, L., & Gunes, H. (2022). Learning multi-dimensional edge feature-based AU relation graph for facial action unit recognition. *arXiv*. https://doi.org/10.48550/arXiv.2205.01782

.. [5] McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In *Proceedings of the 14th Python in Science Conference*, 18-25.

.. [6] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. https://cdn.openai.com/papers/whisper.pdf

.. [4] Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., … Bengio, Y. (2021). SpeechBrain: A general-purpose speech toolkit. *arXiv*. https://doi.org/10.48550/arXiv.2106.04624

.. [1] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *arXiv*. https://doi.org/10.48550/arXiv.1503.03832

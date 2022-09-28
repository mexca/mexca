Features
========

The current release of **mexca** allows users to identify faces and speakers and extract facial expressions, voice features, and spoken text sentiment from video files. The package contains three submodules that process video, audio, and text.

Video
-----

The video processing includes four steps:

1. Detection: Faces displayed in the video frames are detected using a pretrained `MTCNN` model from `facenet-pytorch <https://github.com/timesler/facenet-pytorch>`_ [8]_.
2. Encoding: Faces are extracted from the frames and embeddings are computed using `InceptionResnetV1` from `facenet-pytorch`.
3. Identification: IDs are assigned to faces by clustering the embeddings using spectral clustering.
4. Extraction: Facial features (landmarks, action units) are extracted from the faces using `pyfeat <https://py-feat.org/pages/intro.html>`_ [3]_. Available models are `PFLD`, `MobileFaceNet`, and `MobileNet` for landmark extraction and `JAANET`, `svm`, and `logistic` for action unit extraction.

Audio
-----

The audio processing contains four steps, which are performed by the speaker diarization pipeline from `pyannote.audio <https://github.com/pyannote/pyannote-audio>`_ [2]_:

1. Segmentation: Speech segments from different speakers are detected using `pyannote/segmentation`.
2. Encoding: Speaker embeddings are computed for each speech segment using `ECAPA-TDNN` from `speechbrain <https://speechbrain.github.io/#>`_ [6]_.
3. Identification: IDs are assigned to speech segments based on clustering with a Gaussian hidden Markov model.
4. Extraction: Voice features are extracted using `praat-parselmouth <https://github.com/YannickJadoul/Parselmouth>`_ [5]_. Currently, only the voice pitch (F0) can be extracted.

Text
----

The text is processed in four steps:

1. Transcription: The speech is transcribed into text using a fine-tuned `wave2vec` model [7]_ from `huggingsound <https://github.com/jonatasgrosman/huggingsound>`_. Languages that can be transcribed are English and Dutch.
2. Punctuation: The punctuation in the transcribed text is restored using a pretrained multilingual RoBERTa model from `deepmultilingualpunctuation <https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large>`_ [4]_.
3. Structuring: The transcribed text is structured into sentences using `SpaCy <https://spacy.io/api/sentencizer>`_.
4. Extraction: The sentiment is extracted from the text using a pretrained multilingual RoBERTa `model <https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment>`_ [1]_.

References
----------

.. [1] Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L.. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. *arxiv*. https://doi.org/10.48550/arxiv.2010.12421

.. [2] Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *arXiv*. https://doi.org/10.48550/arXiv.2104.04045

.. [3] Cheong, J. H., Xie, T., Byrne, S., & Chang, L. J. (2021). Py-feat: Python facial expression analysis toolbox. *arXiv*. https://doi.org/10.48550/arXiv.2104.03509

.. [4] Guhr, O., Schumann, A. K., Bahrmann, F., & Böhme, H. (2021). FullStop: Multilingual deep models for punctuation prediction. *Proceedings of the Swiss Text Analytics Conference 2021*. http://ceur-ws.org/Vol-2957/sepp_paper4.pdf 

.. [5] Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

.. [6] Ravanelli, M., Parcollet, T., Plantinga, P., Rouhe, A., Cornell, S., Lugosch, L., … Bengio, Y. (2021). SpeechBrain: A general-purpose speech toolkit. *arXiv*. https://doi.org/10.48550/arXiv.2106.04624

.. [7] Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). wav2vec: Unsupervised pre-training for speech recognition. *arXiv*. https://doi.org/10.48550/arXiv.1904.05862

.. [8] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *arXiv*. https://doi.org/10.48550/arXiv.1503.03832

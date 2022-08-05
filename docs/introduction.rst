Introduction
============

The study of human affective behaviour requires capturing emotion expressions in diverse, naturalistic environments (in the *wild*) from different modalities, such as video, audio and text.
Currently, there is a number of commercial and open-source tools which analyse human emotions in video, audio and text.
However, none of these tools is currently capable of combining state-of-the-art, open-source, and free solutions into one single workflow. For this reason, we developed **mexca**.

*Important*: **Mexca** does not capture emotions, such as "happy" or "sad", but *emotion expressions*. Thus, it does not allow users to make direct inferences about the internal emotional states of persons shown in videos.
Instead, **mexca** is intended to study which emotions humans express (more or less intenionally) to the outside world.

**Mexca** is an easy-to-use, customizable Python package designed to perform facial, speech and text analyses in a single pipeline.
It relies on three main open-source libraries: `Py-Feat <https://py-feat.org/pages/intro.html>`_ for facial expression analysis, `praat-parselmouth <https://github.com/YannickJadoul/Parselmouth>`_ for pitch analysis and `pyannote-audio <https://github.com/pyannote/pyannote-audio>`_ for speaker diarization. Speech-to-text transcription relies on pre-trained models made available and fine-tuned by `huggingsound <https://github.com/jonatasgrosman/huggingsound>`_.

The package is split into three submodules (video, audio and text) which can be used separately. The `core` submodule combines all three modalities and provides the main pipeline.
The modularity of **mexca** allows users to customize their pipeline and easily add new components (e.g., to use more recent pretrained models).

Intended Use
------------

We strongly suggest to apply **mexca** only to videos of public persons or persons that have given explicit consent to analysis of their emotion expressions.
We strongly discourage the use of **mexca** on private videos or for surveillance purposes.
The main intended use case for **mexca** is (open) research.

Licensing
---------

This software is being developed by the `Netherlands eScience Center <https://www.esciencecenter.nl/>`_ in collaboration with the `Hot Politics Lab <http://www.hotpolitics.eu/>`_, at the University of Amsterdam. Code and data are licensed under the Apache License, version 2.0. This means that Mexca can be used, modified and redistributed for free, even for commercial purposes.

References
----------

Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. *arXiv*. https://doi.org/10.48550/arXiv.2104.04045

Cheong, J. H., Xie, T., Byrne, S., & Chang, L. J. (2021). Py-feat: Python facial expression analysis toolbox. *arXiv*. https://doi.org/10.48550/arXiv.2104.03509

Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). wav2vec: Unsupervised pre-training for speech recognition. *arXiv*. https://doi.org/10.48550/arXiv.1904.05862

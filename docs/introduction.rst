Introduction
============

The study of human affective behaviour requires capturing emotions in diverse, naturalistic environments, i.e., in the *wild* -- across video, audio and text. Currently, there is a number of commercially usable machine learning frameworks which provide a way to analyse human emotions in video, audio and text to anyone with a good command of python. However, to date there is no software capable of combining all these state-of-the-art solutions into one single workflow. For this reason, we have designed **Mexca**.


**Mexca** is an easy-to-use python package designed to run facial, speech and text analyses into one single pipeline. It piggybacks on three main open-source libraries: `Py-Feat <https://py-feat.org/pages/intro.html>`_ for facial expression analysis,  `Praat-parselmouth <https://github.com/YannickJadoul/Parselmouth>` for pitch analysis and `Pyannote-audio <https://github.com/pyannote/pyannote-audio>`_ for speaker diarization. Speech-to-text transcription relies on pre-trained models made available and fine-tuned by `Huggingsound <https://github.com/jonatasgrosman/huggingsound>`_.

**Mexca** is split into three main sub-parts (video, audio and text) which are integrated into one single pipeline by the **core** module. Thanks to its modularity, users can autonomously decide whether to run a single pipeline which calls the video, audio and text pipelines, or to run a specific pipeline among the above.

Licensing
---------

This software is being developed by the `Netherlands eScience Center <https://www.esciencecenter.nl/>`_ in collaboration with the `Hot Politics Lab <http://www.hotpolitics.eu/>`, at the University of Amsterdam. Code and data are licensed under the Apache License, version 2.0. This means that Mexca can be used, modified and redistributed for free, even for commercial purposes.



# Multimodal Emotion Expression Capture Amsterdam

[![github license badge](https://img.shields.io/github/license/mexca/mexca)](https://github.com/mexca/mexca)
[![RSD](https://img.shields.io/badge/rsd-mexca-00a3e3.svg)](https://www.research-software.nl/software/mexca)
[![read the docs badge](https://readthedocs.org/projects/pip/badge/)](https://mexca.readthedocs.io/en/latest/index.html)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_mexca&metric=alert_status)](https://sonarcloud.io/dashboard?id=mexca_mexca)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_mexca&metric=coverage)](https://sonarcloud.io/dashboard?id=mexca_mexca)
[![build](https://github.com/mexca/mexca/actions/workflows/build.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/build.yml)
[![cffconvert](https://github.com/mexca/mexca/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/cffconvert.yml)
[![markdown-link-check](https://github.com/mexca/mexca/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/markdown-link-check.yml)

## How to use mexca

This package provides a customizable yet easy-to-use pipeline for extracting emotion expression features from videos. It contains building blocks that can be used to extract features for individual modalities (i.e., facial expressions, voice, and text). The blocks can also be integrated into a single pipeline to extract the features from all modalities at once. Next to extracting features, mexca can also identify the speakers shown in the video by clustering speaker and face representations. This allows users to compare emotion expressions across speakers, time, and contexts.

Currently, mexca includes three independent modules (video, audio and text) which are integrated into a *core* module. The core module is responsible for running a single pipeline which calls in turn all the other sub-modules.

The video submodule supports the extraction of facial features (e.g., facial landmarks, action units). It relies on [pyfeat](https://py-feat.org/pages/intro.html) and it includes the following components [1]:

- Face detection with Multi-task Convolutional Neural Network (MTCNN). F1 score = 0.54
- Landmark detection with PFLD model. RMSE = 6.41
- Action unit detection with Jaa-Net neural network model. F1 score = .22

The audio module relies on [praat-parselmouth](https://github.com/YannickJadoul/Parselmouth) for pitch analysis, and on [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization [2]. It includes the following components:

- Pitch's fundamental frequency (F0)
- Speaker diarization with ECAPA-TDNN model. DER = 3.01

The text module supports text transcriptions for Dutch and English audio files. It relies on a pre-trained model made available by [hugging sound](https://github.com/jonatasgrosman/huggingsound) that is the wav2vec-large model [3] fine-tuned on Dutch and English. WER = 5.10. 

Please cite mexca if you use it for scientific or commercial purposes.

## Getting Started

Emotion expression features can be extracted with mexca using the following lines of code:

```python
from mexca.core.pipeline import Pipeline

# Path to video file (consider using os.path.join())
filename = 'path/to/video'

# Create pipeline object from default constructor method
pipeline = Pipeline().from_default(language='english')

# Apply pipeline to video file (may take a long time depending on video length)
output = pipeline.apply(filename)
```

## Installation

We recommend to install mexca in a new virtual environment, e.g., using `venv`:

```console
python3 -m venv env
env/bin/activate
```

To install mexca from the GitHub repository, do:

```console
git clone https://github.com/mexca/mexca.git
cd mexca
python3 -m pip install .
```

## Documentation

The documentation of mexca can be found on [Read the Docs](https://mexca.readthedocs.io/en/latest/index.html).

## Contributing

If you want to contribute to the development of mexca,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## License

The code is licensed under the Apache 2.0 License. This means that mexca can be used, modified and redistributed for free, even for commercial purposes.

## Credits

Mexca is being developed by the [Netherlands eScience Center](https://www.esciencecenter.nl/) in collaboration with the [Hot Politics Lab](http://www.hotpolitics.eu/) at the University of Amsterdam.

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## References
[1] Cheong, J. H., Xie, T., Byrne, S., & Chang, L. J. (2021). Py-feat: Python facial expression analysis toolbox. arXiv preprint arXiv:2104.03509.

[2] Bredin, H., & Laurent, A. (2021). End-to-end speaker segmentation for overlap-aware resegmentation. arXiv preprint arXiv:2104.04045.

[3] Schneider, S., Baevski, A., Collobert, R., & Auli, M. (2019). wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862.
## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/mexca/mexca) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/mexca/mexca)](https://github.com/mexca/mexca) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-mexca-00a3e3.svg)](https://www.research-software.nl/software/mexca) |
| (4/5) citation                     | |
| (5/5) checklist                    | |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_mexca&metric=alert_status)](https://sonarcloud.io/dashboard?id=mexca_mexca) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=mexca_mexca&metric=coverage)](https://sonarcloud.io/dashboard?id=mexca_mexca) |
| Documentation                      | |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/mexca/mexca/actions/workflows/build.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](https://github.com/mexca/mexca/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/mexca/mexca/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/mexca/mexca/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/mexca/mexca/actions/workflows/markdown-link-check.yml) |

## How to use mexca

This package provides a customizable yet easy-to-use pipeline for extracting emotion expression features from videos. It contains building blocks that can be used to extract features for individual modalities (i.e., facial expressions, voice, and text). The blocks can also be integrated into a single pipeline to extract the features from all modalities at once. Next to extracting features, mexca can also identify the speakers shown in the video by clustering speaker and face representations. This allows users to compare emotion expressions across speakers, time, and situations.

Currently, mexca supports the extraction of the following features:
- Facial expressions (using [pyfeat](https://py-feat.org/pages/intro.html))
  - Facial landmarks
  - Facial action units
- Voice (using [praat-parselmouth](https://github.com/YannickJadoul/Parselmouth))
  - Pitch (F0)

Please cite mexca if you use it for scientific purposes.

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

## Contributing

If you want to contribute to the development of mexca,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

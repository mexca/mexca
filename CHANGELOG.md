# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1-beta] - 2023-02-03

Minor patch that addresses a memory issue and includes some bug and documentation fixes.

### Added

- A "Troubleshooting" sub section in the "Installation Details" section in docs.
- Exception class `AuthenticationError` for failed HuggingFace Hub authentication
- Exception class `NotEnoughFacesError` for too few face detections for clustering

### Changed

- Refactored `VideoDataset` class to only load video frames when they are queried. The previous implementation attemped to load the entire video into memory leading to issues. Now, only frames of the current batch are loaded into memory as expected.

### Fixed

- Added missing note about HuggingFace Hub authentication to "Getting Started" section in docs.
- An exception is triggered if pypiwin32 was not properly installed when initializing a docker client
- An exception is triggered if no HuggingFace Hub token was found when initializing `SpeakerIdentifier` with `use_auth_token=True`
- Correctly passes the HuggingFace Hub token to the Docker build action for the SpeakerIdentifier container

## [0.2.0-beta] - 2023-01-26

First beta release. This version is a major overhaul of the first alpha release.

### Added

- A component for sentiment extraction
- Data classes as interfaces for component in- and output in the `data` module
- CLIs for all five components, removes general CLI for pipeline
- Interfaces for Docker containers of all five components, removes general Dockerfile
- Functionality to write output to common file formats (JSON, RTTM, SRT)
- Lazy initialization for pretrained models to save memory
- Data loader functionality to the `FaceExtractor` component to allow for batch processing
- Clustering confidence metric to the output of the `FaceExtractor` class
- Logging
- Static type annotations
- Added utils module
- Flowchart to the introduction in docs
- 'Getting Started' section in docs

### Changed

- Simplified the structure of the package
- Moved content of core module into separate modules
- Refactors the `Pipeline` class to include five components: `FaceExtractor`, `SpeakerIdentifier`, `VoiceExtractor`, `AudioTranscriber`, `SentimentExtractor`
- Separated the dependencies for all five components: They can all be installed separately from each other
- Whisper for audio transcription instead of fine-tuned wav2vec models via huggingsound
- Adapted the `FaceExtractor` component for the pretrained models used in py-feat v0.5
- Refactors feature merging using `pandas.DataFrame` and `intervaltree.IntervalTree`
- Splits the installation instructions in two parts (quick vs. detailed) in docs
- Updates docker section
- Updates command line section

### Removed

- Removed the `AudioIntegrator` and `AudioTextIntegrator` classes, feature merging is done in the `Multimodal` class
- Removed the core module and its submodules
- Removed face-speaker matching (temporarily); might be added again in a future release

## [0.1.0-alpha] - 2022-08-09

First alpha release.

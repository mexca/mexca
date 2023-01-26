# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-beta] - 2022-01-26

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

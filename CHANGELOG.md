# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## v0.1.0-alpha

First alpha release.

## v0.2.0-beta

First beta release. This version is a major overhaul of the first alpha release: It simplifies the package structure and removes the `AudioIntegrator` and `AudioTextIntegrator` classes. It also uses
Whisper for audio transcription instead of fine-tuned wav2vec models via huggingsound. Furthermore, it
adds a component for sentiment extraction. Further major changes are:

- Simplifies the structure of the package, removes the core module and moves it's contents into separate modules
- Adds a component for sentiment extraction
- Separates the dependencies for all five components: They can all be installed separately from each other
- Uses Whisper for audio transcription instead of fine-tuned wav2vec models via huggingsound
- Adds a data loader functionality to the `FaceExtractor` component to allow for batch processing
- Adapts the `FaceExtractor` component for the pretrained models used in py-feat v0.5
- Adds a clustering confidence metric to the output of the `FaceExtractor` class
- Removes the `AudioIntegrator` and `AudioTextIntegrator` classes, feature merging is done in the `Multimodal` class
- Refactors the `Pipeline` class to include five components: `FaceExtractor`, `SpeakerIdentifier`, `VoiceExtractor`, `AudioTranscriber`, `SentimentExtractor`
- Adds CLIs for all five components, removes general CLI for pipeline
- Refactors feature merging using `pandas.DataFrame` and `intervaltree.IntervalTree`
- Adds data classes as interfaces for component in- and output in the `data` module
- Adds interfaces for Docker containers of all five components, removes general Dockerfile
- Adds functionality to write output to common file formats (JSON, RTTM, SRT)
- Adds lazy initialization for pretrained models to save memory
- Adds logging
- Adds static type annotations

The following changes were made to the documentation:
- Splits the installation instructions in two parts (quick vs. detailed)
- Adds a flowchart to the introduction
- Updates docker section
- Updates command line section
- Adds 'Getting Started' section

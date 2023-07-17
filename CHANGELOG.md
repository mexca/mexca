# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0-beta] - 2023-07-17

Replaces the methods for predicting facial landmarks and action unit activations. Landmarks are now predicted by facenet-pytorch and action units by the MEFARG model instead of py-feat.

### Added

- The `FaceExtractor` component computes average face embeddings via the `compute_avg_embeddings()` method
- The `VideoAnnotation` data class has an additional field `face_average_embeddings`, containing the average face representations for each detected face cluster
- The `AudioTranscriber` component returns the confidence of the transcription (average over each word sequence)
- The `TranscriptionData` data class has an additional field `confidence` for the transcription confidence
- `ruamel.yaml` is added as an explicit dependency for the SpeakerIdentifier component
- `gdown` is added as a dependency for the FaceExtractor component
- Package code adheres to black code style
- Adds pre-commit configuration (enable with `pre-commit install`) in `.pre-commit-config.yaml`
- Adds `black` and `pre-commit` to .\[dev\] requirements

### Changed

- The `mexca.video` module is split into several submodules: `extraction`, `mefl`, `anfl`, `mefarg`, and `helper_classes`
- Facial landmarks are predicted by `facenet_pytorch.MTCNN` instead of `feat.detector.Detector`
- Facial action units are predicted by `mexca.video.mefarg.MEFARG` instead of `feat.detector.Detector`
- Word-level timestamps are obtained from the native `whisper` package instead of `stable-ts`

### Removed

- `py-feat` is removed from the dependencies of the FaceExtractor component
- `stable-ts` is removed from the dependencies of the AudioTranscriber component


## [0.4.0-beta] - 2023-04-26

Adds voice features, improves the documentation, and updates the example notebooks. Enables GPU support for all pipeline components.

### Added

- Classes for extracting voice features in the `mexca.audio.features` module:
    - `FormantAudioSignal` for preemphasized audio signals for formant analysis
    - `AlphaRatioFrames` and `HammarIndexFrames` for calculating alpha ratio and Hammarberg index
    - `SpectralSlopeFrames` for estimating spectral slopes
    - `MelSpecFrames` and `MfccFrames` for computing Mel spectrograms and cepstral coefficients
    - `SpectralFluxFrames` and `RmsEnergyFrames` for calculating spectral flux and RMS energy
- Classes for extracting and interpolating voice features in the `mexca.audio.extraction` module: `FeatureAlphaRatio`, `FeatureHammarIndex`, `FeatureSpectralSlope`, `FeatureHarmonicDifference`, `FeatureMfcc`, `FeatureSpectralFlux`, `FeatureRmsEnergy`
- A `VoiceFeaturesConfig` class for configuring voice feature extraction in `mexca.data`
- The CLI `extract-voice` has a new `--config-filepath` argument for YAML configuration files
- The `FaceExtractor` component has new `max_cluster_frames` argument to set the maximum number of frames for spectral clustering
- The `SentimentExtractor` component has a new `device` argument to run on GPU with 8-bit
- `pyyaml` is added as a requirement for the base package
- `accelerate` and `bisandbytes` are added as requirements for the SentimentExtractor component

### Changed

- The set of default voice features that are extracted with `VoiceExtractor` has been expanded
- The default window for STFT is now the Hann window
- Conversion from magnitude/energy to dB is now performed with librosa functions
- The example notebooks have been updated
- `mexca.container.VoiceExtractorContainer` can also handle `VoiceFeaturesConfig` objects
- Required version of `spectralcluster` is set to 0.2.16
- Required version of `transformers` is set to 4.25.1
- Required version of numpy is set to >=1.21.6

### Fixed

- Warnings triggered during voice feature extraction
- A bug occuring when using `FaceExtractor` with `device="cuda"` has been fixed


## [0.3.0-beta] - 2023-04-05

Improves the audio transcription and sentiment extraction workflows. Refactors the voice feature extraction workflow and adds several new voice features.

### Added

- Docker containers are now versioned via tags and the container components automatically fetch the container matching the installed version of mexca; the container with the `:latest` tag can be fetched with the argument `get_latest_tag=True` (#65)
- Classes for extracting voice features (#66):
    - `AudioSignal`, `BaseSignal` for loading and storing signals in the `mexca.audio.features` module
    - `BaseFrames`, `FormantFrames`, `FormantAmplitudeFrames`, `HnrFrames`, `JitterFrames`, `PitchFrames`, `PitchHarmonicsFrames`, `PitchPeriodFrames`, `PitchPulseFrames`, `ShimmerFrames`, `SpecFrames` for computing and storing formant features, glottal pulse features, and pitch features in the `mexca.audio.features` module
    - `BaseFeature`, `FeaturePitchF0`, `FeatureJitter`, `FeatureShimmer`, `FeatureHnr`, `FeatureFormantFreq`, `FeatureFormantBandwidth`, `FeatureFormantAmplitude` for extracting and interpolating voice features in the `mexca.audio.extraction` module
- An `all` extra requirements group which installs the requirements for all of mexca's components (i.e., `pip install mexca[all]`, #64)

### Changed

- The `SentimentData` class now has a `text` instead of an `index` attribute, which is used for matching sentiment to transcriptions (#63)
- The sentence sentiment is merged separately from the transcription in `Multimodal._merge_audio_text_features()` (#63)
- librosa (version 0.9) is added as a requirement for the VoiceExtractor component instead of parselmouth; the voice feature extraction now relies on librosa instead of praat (#66)
- stable-ts is required to be version 1.1.5 for compatibility with Python 3.7; in a future version, we might remove stable-ts as a dependency (#67)
- transformers is added as a requirement for the AudioTranscriber component (#67)
- scipy is moved to the general requirements for all components (#66)
- The `VoiceExtractor` class and component is refactored with new default features (#66)
- Tests make better use of fixtures for cleaner and more reusable code (#63)

### Fixed
- An error in the audio transcription that occurred for extremely short speech segments below the precision of whisper and stable-ts (#63)

### Removed
- The `toml` extra requirement for the coverage requirement in the `dev` group (#67)

## [0.2.1-beta] - 2023-02-03

Minor patch that addresses a memory issue and includes some bug and documentation fixes.

### Added

- A "Troubleshooting" sub section in the "Installation Details" section in docs
- Exception class `AuthenticationError` for failed HuggingFace Hub authentication
- Exception class `NotEnoughFacesError` for too few face detections for clustering

### Changed

- Refactored `VideoDataset` class to only load video frames when they are queried. The previous implementation attemped to load the entire video into memory leading to issues. Now, only frames of the current batch are loaded into memory as expected.

### Fixed

- Added missing note about HuggingFace Hub authentication to "Getting Started" section in docs
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

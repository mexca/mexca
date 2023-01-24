Command Line
============

The mexca package contains multiple CLIs which can be called to run the pipeline components. The documentation of each CLI can be seen via:

.. code-block:: console

  entrypoint --help # e.g., extract-faces --help

The five entrypoints are:

- `extract-faces`: Applies the FaceExtractor component to a video file.
- `identify-speakers`: Applies the SpeakerIdentifier component to an audio file.
- `extract-voice`: Applies the VoiceExtractor component to an audio file.
- `transcribe`: Applies the AudioTranscriber component to an audio file taking detected speech segments into account.
- `extract-sentiment`: Applies the SentimentExtractor component to a transcription file.

.. note::
  The CLIs are currently mostly for internal use in the containerized components.

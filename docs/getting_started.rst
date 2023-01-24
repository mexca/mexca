Getting Started
===============

This section gives a quick overview on how to use the mexca package. For detailed examples, check out the `example <https://github.com/mexca/mexca/tree/main/examples>`_ notebooks.

To create and run the MEXCA pipeline on a video file run the following code in a Jupyter notebook or a Python script:

.. code-block:: python

    from mexca.audio import SpeakerIdentifier, VoiceExtractor
    from mexca.data import Multimodal
    from mexca.pipeline import Pipeline
    from mexca.text import AudioTranscriber, SentimentExtractor
    from mexca.video import FaceExtractor

    # Set path to video file
    filepath = 'path/to/video'

    # Create standard pipeline with two faces and speakers
    pipeline = Pipeline(
        face_extractor=FaceExtractor(num_faces=2),
        speaker_identifier=SpeakerIdentifier(
            num_speakers=2,
            use_auth_token=True
        ),
        voice_extractor=VoiceExtractor(),
        audio_transcriber=AudioTranscriber(),
        sentiment_extractor=SentimentExtractor()
    )

    # Apply pipeline to video file at `filepath`
    result = pipeline.apply(
        filepath,
        frame_batch_size=5,
        skip_frames=5
    )

    # Print merged features
    print(result.features)

The result should be a pandas data frame printed to the console or notebook output.
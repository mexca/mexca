Running mexca on a GPU
======================

For real-world applications, it is recommended to run mexca with GPU support as this substantially accelerates the computation performance.
Running mexca on a GPU requires the CUDA toolkit and PyTorch with CUDA to be installed (see :ref:`installation with cuda`).
The ``device`` argument of the :class:`FaceExtractor`, :class:`AudioTranscriber`, and :class:`SentimentExtractor` components enables GPU support.
The :class:`SpeakerIdentifier` component automatically detects whether a GPU is available for use.

.. code-block:: python

    import torch

    from mexca.audio import SpeakerIdentifier, VoiceExtractor
    from mexca.data import Multimodal
    from mexca.pipeline import Pipeline
    from mexca.text import AudioTranscriber, SentimentExtractor
    from mexca.video import FaceExtractor

    # Set path to video file
    filepath = 'path/to/video'

    # Run with CUDA if available
    device = (
        torch.device(type="cuda")
        if torch.cuda.is_available()
        else torch.device(type="cpu")
    )

    # Create standard pipeline with two faces and speakers
    pipeline = Pipeline(
        face_extractor=FaceExtractor(num_faces=2, device=device),
        speaker_identifier=SpeakerIdentifier(
            num_speakers=2,
            device=device,
            use_auth_token=True # login with token required
        ),
        voice_extractor=VoiceExtractor(),
        audio_transcriber=AudioTranscriber(device=device),
        sentiment_extractor=SentimentExtractor(device=device)
    )

    # Apply pipeline to video file at `filepath`
    result = pipeline.apply(
        filepath,
        frame_batch_size=5,
        skip_frames=5
    )

    # Print merged features
    print(result.features.collect())

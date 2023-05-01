from mexca.text import AudioTranscriber, SentimentExtractor
from mexca.audio import SpeakerIdentifier, VoiceExtractor
from mexca.video import FaceExtractor
from mexca.pipeline import Pipeline

# Set path to video file
filepath = '../tests/test_files/test_video_audio_5_seconds.mp4'

# Create standard pipeline with two faces and speakers
pipeline = Pipeline(
    face_extractor=FaceExtractor(num_faces=1),
    speaker_identifier=SpeakerIdentifier(
        num_speakers=1,
        use_auth_token='[authorisation token]'
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
"""Script for creating reference files for testing.
"""

import os
from mexca.audio import VoiceExtractor
from mexca.pipeline import Pipeline
from mexca.video import FaceExtractor
# from moviepy.editor import VideoFileClip

# with VideoFileClip('examples/debate.mp4') as clip:
#     clip.subclip(5, 10).write_videofile(os.path.join(
#         'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
#     ))

video_filepath = os.path.join(
    'tests', 'test_files', 'test_video_audio_5_seconds.mp4'
)

video_reference_path = os.path.join(
    'tests', 'reference_files', 'test_video_audio_5_seconds_video_annotation.json'
)

face_extractor = FaceExtractor(num_faces=4)

video_annotation = face_extractor.apply(video_filepath, batch_size=5, skip_frames=5)

video_annotation.write_json(video_reference_path)

audio_filepath = os.path.join(
    'tests', 'test_files', 'test_video_audio_5_seconds.wav'
)

audio_reference_path = os.path.join(
    'tests', 'reference_files', 'test_video_audio_5_seconds_voice_features.json'
)

extractor = VoiceExtractor()

pipeline = Pipeline(voice_extractor=extractor)

features = pipeline.apply(filepath=video_filepath, skip_frames=5)

features.voice_features.write_json(audio_reference_path)

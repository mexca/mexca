"""Containers of pipeline components.
"""

import os
from typing import List, Optional, Tuple, Union
import docker
from docker.types import Mount
from mexca.data import AudioTranscription, SentimentAnnotation, SpeakerAnnotation, VideoAnnotation, VoiceFeatures


class BaseContainer:
    """Base class for container components. Only for internal use.
    """
    def __init__(self, image_name: str):
        self.image_name = image_name
        self.client = docker.from_env()
        self.mount_dir = '/mnt/vol'

        try:
            self.client.images.get(image_name)
        except docker.errors.ImageNotFound:
            self.client.images.pull(image_name)


    def _create_mounts(self, filepath: str):
        outdir = os.path.abspath(os.path.dirname(filepath))
        self.mounts = [
            Mount(
                target=self.mount_dir,
                source=outdir,
                type='bind'
            )
        ]

        return outdir


    @staticmethod
    def _create_out_path_stem(filepath: str, outdir: str):
        return os.path.join(outdir, os.path.splitext(os.path.basename(filepath))[0])


    @staticmethod
    def _create_base_cmd(filepath: str) -> List[str]:
        mount_dir = '../mnt/vol/'
        return ['-f', mount_dir + os.path.basename(filepath), '-o', mount_dir]


    def _run_container(self, args: List[str], show_progress: bool = True):
        container = self.client.containers.run(
            self.image_name,
            args,
            remove=True,
            detach=True,
            mounts=self.mounts
        )

        if show_progress:
            for s in container.attach(stream=True):
                print(s.decode('utf-8'))

        container.wait()


class FaceExtractorContainer(BaseContainer):
    """Container for `FaceExtractor` component.

    Other Parameters
    ----------------
    image_name: str, default='mexca-face-extractor'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    FaceExtractor

    """
    def __init__(self,
        num_faces: Optional[int],
        min_face_size: int = 20,
        thresholds: Tuple[float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        selection_method: Optional[str] = None,
        keep_all: bool = True,
        device: Optional['torch.device'] = 'cpu',
        embeddings_model: str = 'vggface2',
        au_model: str = 'xgb',
        landmark_model: str = 'mobilefacenet',
        image_name: str = 'mexca/face-extractor'
    ):
        self.num_faces = num_faces
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.selection_method = selection_method
        self.keep_all = keep_all
        self.device = device
        self.embeddings_model = embeddings_model
        self.au_model = au_model
        self.landmark_model = landmark_model

        super().__init__(image_name=image_name)

        
    def apply(self,
        filepath: str,
        batch_size: int = 1,
        skip_frames: int = 1,
        process_subclip: Tuple[Optional[float]] = (0, None),
        show_progress: bool = True
    ) -> VideoAnnotation:

        cmd_args = [
            '--num-faces', self.num_faces,
            '--batch-size', batch_size,
            '--skip-frames', skip_frames,
            '--process-subclip', process_subclip[0], process_subclip[1],
            '--show-progress', show_progress,
            '--min-face-size', self.min_face_size,
            '--thresholds', self.thresholds[0], self.thresholds[1], self.thresholds[2],
            '--factor', self.factor,
            '--post-process', self.post_process,
            '--select-largest', self.select_largest,
            '--selection-method', self.selection_method,
            '--keep-all', self.keep_all,
            '--device', self.device,
            '--embeddings-model', self.embeddings_model,
            '--au-model', self.au_model,
            '--landmark-model', self.landmark_model 
        ]

        # Convert cli args to string (otherwise docker entrypoint can't read them)
        cmd_args_str = [str(arg) for arg in cmd_args]

        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args_str)

        return VideoAnnotation.from_json(self._create_out_path_stem(filepath=filepath, outdir=outdir) + '_video_annotation.json') 


class SpeakerIdentifierContainer(BaseContainer):
    """Container for `SpeakerIdentifier` component.

    Other Parameters
    ----------------
    image_name: str, default='mexca-speaker-identifier'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    SpeakerIdentifier

    """
    def __init__(self,
        num_speakers: Optional[int] = None,
        use_auth_token: Union[bool, str] = True,
        image_name: str = 'mexca/speaker-identifier'
    ):
        self.num_speakers = num_speakers
        self.use_auth_token = use_auth_token

        super().__init__(image_name=image_name)


    def apply(self, filepath: str) -> SpeakerAnnotation:
        cmd_args = [
            '--num-speaker', str(self.num_speakers),
            '--use-auth-token', str(self.use_auth_token)
        ]
        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args)

        return SpeakerAnnotation.from_rttm(self._create_out_path_stem(filepath=filepath, outdir=outdir) + '_audio_annotation.rttm')


class VoiceExtractorContainer(BaseContainer):
    """Container for `VoiceExtractor` component.

    Other Parameters
    ----------------
    image_name: str, default='mexca-voice-extractor'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    VoiceExtractor

    """
    def __init__(self, image_name: str = 'mexca/voice-extractor'):
        super().__init__(image_name=image_name)


    def apply(self, filepath: str, time_step: float, skip_frames: int = 1) -> VoiceFeatures:
        cmd_args = ['--time-step', str(time_step), '--skip-frames', str(skip_frames)]
        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args)

        return VoiceFeatures.from_json(self._create_out_path_stem(filepath=filepath, outdir=outdir) + '_voice_features.json')


class AudioTranscriberContainer(BaseContainer):
    """Container for `AudioTrascriber` component.

    Other Parameters
    ----------------
    image_name: str, default='mexca-audio-transcriber'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    AudioTranscriber

    """
    def __init__(self,
        whisper_model: Optional[str] = 'small',
        device: Optional[Union[str, 'torch.device']] = 'cpu',
        sentence_rule: Optional[str] = None,
        image_name: str = 'mexca/audio-transcriber'
    ):
        self.whisper_model = whisper_model
        self.device = device
        self.sentence_rule = sentence_rule

        super().__init__(image_name=image_name)


    def apply(self,
        filepath: str,
        _, # audio_annotation in AudioTranscriber.apply()
        language: Optional[str] = None,
        show_progress: bool = True
    ) -> AudioTranscription:
        cmd_args = [
            '--show-progress', str(show_progress),
            '--annotation-path', '../mnt/vol/' + os.path.splitext(os.path.basename(filepath))[0] + '_audio_annotation.rttm',
            '--language', str(language)
        ]
        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args)

        transcription = AudioTranscription.from_srt(self._create_out_path_stem(filepath=filepath, outdir=outdir) + '_transcription.srt')

        return transcription


class SentimentExtractorContainer(BaseContainer):
    """Container for `SentimentExtractor` component.

    Other Parameters
    ----------------
    image_name: str, default='mexca-sentiment-extractor'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    SentimentExtractor

    """
    def __init__(self, image_name: str = 'mexca/sentiment-extractor'):
        super().__init__(image_name=image_name)


    def apply(self, transcription: AudioTranscription, show_progress: bool = True) -> SentimentAnnotation:
        cmd_args = [
            '--transcription-path', '../mnt/vol/' + os.path.basename(transcription.filename),
            '--outdir', '../mnt/vol',
            '--show-progress', str(show_progress),
        ]
    
        outdir = self._create_mounts(filepath=transcription.filename)

        self._run_container(cmd_args)

        base_dir = '_'.join(self._create_out_path_stem(filepath=transcription.filename, outdir=outdir).split('_')[:-1])

        sentiment = SentimentAnnotation.from_json(base_dir + '_sentiment.json')

        return sentiment

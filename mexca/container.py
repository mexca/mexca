"""Containers of pipeline components.
"""

import os
from typing import List, Optional, Tuple, Union

import docker
from docker.errors import DockerException
from docker.types import Mount

from mexca import __version__ as VERSION
from mexca.data import (
    AudioTranscription,
    SentimentAnnotation,
    SpeakerAnnotation,
    VideoAnnotation,
    VoiceFeatures,
    VoiceFeaturesConfig,
)


class BaseContainer:
    """Base class for container components. Only for internal use.

    Parameters
    ----------
    image_name: str
        Name of docker image. A tag is automatically added from the package version.
    get_latest_tag : bool, default=False
        Whether to pull the latest version of the container instead of the version matching the package version.
        This is mainly useful for debugging.

    """

    mounts: Optional[None] = None

    def __init__(self, image_name: str, get_latest_tag: bool = False):
        if get_latest_tag:
            self.image_name = image_name + ":latest"
        else:
            self.image_name = image_name + ":v" + str(VERSION)
        try:
            self.client = docker.from_env()
        except DockerException as exc:
            raise DockerException(
                "pypiwin32 package not correctly installed; running 'python pywin32_postinstall.py -install' might fix this issue"
            ) from exc

        self.mount_dir = "/mnt/vol"

        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            self.client.images.pull(self.image_name)

    def _create_mounts(self, filepath: str):
        outdir = os.path.abspath(os.path.dirname(filepath))
        self.mounts = [Mount(target=self.mount_dir, source=outdir, type="bind")]

        return outdir

    @staticmethod
    def _create_out_path_stem(filepath: str, outdir: str):
        return os.path.join(
            outdir, os.path.splitext(os.path.basename(filepath))[0]
        )

    @staticmethod
    def _create_base_cmd(filepath: str) -> List[str]:
        mount_dir = "../mnt/vol/"
        return ["-f", mount_dir + os.path.basename(filepath), "-o", mount_dir]

    def _run_container(self, args: List[str], show_progress: bool = True):
        container = self.client.containers.run(
            self.image_name, args, detach=True, mounts=self.mounts
        )

        if show_progress:
            for s in container.attach(stream=True):
                print(s.decode("utf-8"))

        exit_code = container.wait()

        if exit_code["StatusCode"] != 0:
            err_msg = container.attach(logs=True)
            raise DockerException(
                "Container returned exit status code - error occurred during component run: "
                + err_msg.decode("utf-8")
            )

        container.remove()

    @staticmethod
    def _remove_output(filepath: str):
        os.remove(filepath)


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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_faces: Optional[int],
        min_face_size: int = 20,
        thresholds: Tuple[float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        selection_method: Optional[str] = None,
        keep_all: bool = True,
        device: Optional["torch.device"] = "cpu",
        max_cluster_frames: Optional[int] = None,
        embeddings_model: str = "vggface2",
        image_name: str = "mexca/face-extractor",
        get_latest_tag: bool = False,
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
        self.max_cluster_frames = max_cluster_frames
        self.embeddings_model = embeddings_model

        super().__init__(image_name=image_name, get_latest_tag=get_latest_tag)

    def apply(
        self,
        filepath: str,
        batch_size: int = 1,
        skip_frames: int = 1,
        process_subclip: Tuple[Optional[float]] = (0, None),
        show_progress: bool = True,
    ) -> VideoAnnotation:
        cmd_args = [
            "--num-faces",
            self.num_faces,
            "--batch-size",
            batch_size,
            "--skip-frames",
            skip_frames,
            "--process-subclip",
            process_subclip[0],
            process_subclip[1],
            "--show-progress",
            show_progress,
            "--min-face-size",
            self.min_face_size,
            "--thresholds",
            self.thresholds[0],
            self.thresholds[1],
            self.thresholds[2],
            "--factor",
            self.factor,
            "--post-process",
            self.post_process,
            "--select-largest",
            self.select_largest,
            "--selection-method",
            self.selection_method,
            "--keep-all",
            self.keep_all,
            "--device",
            self.device,
            "--max-cluster-frames",
            self.max_cluster_frames,
            "--embeddings-model",
            self.embeddings_model,
        ]

        # Convert cli args to string (otherwise docker entrypoint can't read them)
        cmd_args_str = [str(arg) for arg in cmd_args]

        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args_str)

        outpath = (
            self._create_out_path_stem(filepath=filepath, outdir=outdir)
            + f"_{VideoAnnotation.serialization_name()}.json"
        )

        result = VideoAnnotation.from_json(outpath, extra_filename=filepath)

        self._remove_output(outpath)

        return result


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

    def __init__(
        self,
        num_speakers: Optional[int] = None,
        use_auth_token: Union[bool, str] = True,
        image_name: str = "mexca/speaker-identifier",
        get_latest_tag: bool = False,
    ):
        self.num_speakers = num_speakers
        self.use_auth_token = use_auth_token

        super().__init__(image_name=image_name, get_latest_tag=get_latest_tag)

    def apply(self, filepath: str) -> SpeakerAnnotation:
        cmd_args = [
            "--num-speaker",
            str(self.num_speakers),
            "--use-auth-token",
            str(self.use_auth_token),
        ]
        cmd = self._create_base_cmd(filepath=filepath)

        outdir = self._create_mounts(filepath=filepath)

        self._run_container(cmd + cmd_args)

        outpath = (
            self._create_out_path_stem(filepath=filepath, outdir=outdir)
            + f"_{SpeakerAnnotation.serialization_name()}.json"
        )

        result = SpeakerAnnotation.from_json(outpath, extra_filename=filepath)

        self._remove_output(outpath)

        return result


class VoiceExtractorContainer(BaseContainer):
    """Container for `VoiceExtractor` component.

    Other Parameters
    ----------------
    config: VoiceFeaturesConfig, optional, default=None
        Voice feature extraction configuration object. If `None`, uses :class:`VoiceFeaturesConfig`'s default configuration.
    image_name: str, default='mexca-voice-extractor'
        Name of the image to create a container from.
        Pulls the image from Docker Hub if not found locally.

    See Also
    --------
    VoiceExtractor

    """

    def __init__(
        self,
        config: Optional[VoiceFeaturesConfig] = None,
        image_name: str = "mexca/voice-extractor",
        get_latest_tag: bool = False,
    ):
        self.config = config
        super().__init__(image_name=image_name, get_latest_tag=get_latest_tag)

    def apply(
        self, filepath: str, time_step: float, skip_frames: int = 1
    ) -> VoiceFeatures:
        outdir = self._create_mounts(filepath=filepath)

        cmd_args = [
            "--time-step",
            str(time_step),
            "--skip-frames",
            str(skip_frames),
        ]

        if self.config is not None:
            config_path = (
                self._create_out_path_stem(filepath, outdir)
                + "_voice_features_config.yml"
            )
            self.config.write_yaml(config_path)
            cmd_args.extend(
                [
                    "--config-filepath",
                    "../mnt/vol/" + os.path.basename(config_path),
                ]
            )

        cmd = self._create_base_cmd(filepath=filepath)

        self._run_container(cmd + cmd_args)

        if self.config is not None:
            self._remove_output(config_path)

        outpath = (
            self._create_out_path_stem(filepath=filepath, outdir=outdir)
            + f"_{VoiceFeatures.serialization_name()}.json"
        )

        result = VoiceFeatures.from_json(
            filename=outpath, extra_filename=filepath
        )

        self._remove_output(outpath)

        return result


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

    def __init__(
        self,
        whisper_model: Optional[str] = "small",
        device: Optional[Union[str, "torch.device"]] = "cpu",
        sentence_rule: Optional[str] = None,
        image_name: str = "mexca/audio-transcriber",
        get_latest_tag: bool = False,
    ):
        self.whisper_model = whisper_model
        self.device = device
        self.sentence_rule = sentence_rule

        super().__init__(image_name=image_name, get_latest_tag=get_latest_tag)

    def apply(
        self,
        filepath: str,
        audio_annotation: SpeakerAnnotation,  # audio_annotation in AudioTranscriber.apply()
        language: Optional[str] = None,
        show_progress: bool = True,
    ) -> AudioTranscription:
        outdir = self._create_mounts(filepath=filepath)

        annotation_filename = (
            self._create_out_path_stem(filepath, outdir)
            + f"_{SpeakerAnnotation.serialization_name()}.json"
        )
        audio_annotation.write_json(annotation_filename)

        cmd_args = [
            "--show-progress",
            str(show_progress),
            "--annotation-path",
            "../mnt/vol/" + os.path.basename(annotation_filename),
            "--language",
            str(language),
        ]
        cmd = self._create_base_cmd(filepath=filepath)

        self._run_container(cmd + cmd_args)

        outpath = (
            self._create_out_path_stem(filepath=filepath, outdir=outdir)
            + f"_{AudioTranscription.serialization_name()}.json"
        )

        transcription = AudioTranscription.from_json(
            outpath, extra_filename=filepath
        )

        self._remove_output(outpath)
        self._remove_output(annotation_filename)

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

    def __init__(
        self,
        image_name: str = "mexca/sentiment-extractor",
        get_latest_tag: bool = False,
    ):
        super().__init__(image_name=image_name, get_latest_tag=get_latest_tag)

    def apply(
        self, transcription: AudioTranscription, show_progress: bool = True
    ) -> SentimentAnnotation:
        outdir = self._create_mounts(filepath=transcription.filename)

        filepath_split = os.path.splitext(
            os.path.basename(transcription.filename)
        )

        if filepath_split[-1] == ".json":
            transcription_filename = (
                self._create_out_path_stem(transcription.filename, outdir)
                + ".json"
            )
        elif filepath_split[-1] in (".wav", ".mp3"):
            transcription_filename = (
                self._create_out_path_stem(transcription.filename, outdir)
                + f"_{AudioTranscription.serialization_name()}.json"
            )
        else:
            raise ValueError(
                "Object 'transcription' must have a 'filename' attribute pointing to a .json or audio file"
            )

        transcription.write_json(transcription_filename)

        cmd_args = [
            "--transcription-path",
            "../mnt/vol/" + os.path.basename(transcription_filename),
            "--outdir",
            "../mnt/vol",
            "--show-progress",
            str(show_progress),
        ]

        self._run_container(cmd_args)

        outpath = (
            "_".join(
                self._create_out_path_stem(
                    filepath=transcription_filename, outdir=outdir
                ).split("_")[:-1]
            )
            + f"_{SentimentAnnotation.serialization_name()}.json"
        )

        sentiment = SentimentAnnotation.from_json(
            outpath, extra_filename=transcription_filename
        )

        self._remove_output(outpath)
        self._remove_output(transcription_filename)

        return sentiment

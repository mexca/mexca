"""Extract voice features from an audio file.
"""

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import librosa
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from scipy.signal.windows import get_window
from mexca.data import VoiceFeatures
from mexca.utils import ClassInitMessage


class BaseSignal:
    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None


    def __init__(self, sig: np.ndarray, sr: float) -> None:
        self.logger = logging.getLogger('mexca.audio.extraction.BaseSignal')
        self.sig = sig
        self.sr = sr
        self.logger.debug(ClassInitMessage())


    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(self.sig.shape[0])
        return self._idx
    

    @property
    def ts(self) -> np.ndarray:
        if self._ts is None:
            self._ts = librosa.samples_to_time(self.idx, sr=self.sr)
        return self._ts


class BaseFrames:
    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None


    def __init__(self, frames: np.ndarray, sr: int,  frame_len: int, hop_len: int, center: bool = True, pad_mode: str = 'constant') -> None:
        self.logger = logging.getLogger('mexca.audio.extraction.BaseFrames')
        self.frames = frames
        self.sr = sr
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.center = center
        self.pad_mode = pad_mode
        self.logger.debug(ClassInitMessage())


    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(self.frames.shape[0])
        return self._idx
    

    @property
    def ts(self) -> np.ndarray:
        if self._ts is None:
            self._ts = librosa.frames_to_time(self.idx, sr=self.sr, hop_length=self.hop_len)
        return self._ts


class PitchFrames(BaseFrames):
    def __init__(self, frames: np.ndarray, flag: np.ndarray, prob: np.ndarray, sr: int, 
                 lower: float, upper: float, frame_len: int, hop_len: int, method: str,
                 center: bool = True, pad_mode: str = 'constant'):
        self.logger = logging.getLogger('mexca.audio.extraction.PitchFrames')
        self.flag = flag
        self.probs = prob
        self.lower = lower
        self.upper = upper
        self.method = method
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())


class AudioSignal(BaseSignal):
    def __init__(self, sig: np.ndarray, sr: float, filename: str, mono: bool) -> None:
        self.logger = logging.getLogger('mexca.audio.extraction.AudioSignal')
        self.filename = filename
        self.mono = mono
        super().__init__(sig, sr)
        self.logger.debug(ClassInitMessage())


    @classmethod
    def from_file(cls, filename: str, sr: Optional[int] = None, mono: bool = True):
        sig, nat_sr = librosa.load(path=filename, sr=sr, mono=mono)
        return cls(sig, nat_sr, filename, mono)
    

    def _calc_pitch_pyin(self, frame_len: int, hop_len: int, lower: float = 75.0, upper: float = 600.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return librosa.pyin(self.sig, fmin=lower, fmax=upper, sr=self.sr, frame_length=frame_len, hop_length=hop_len)


    def to_pitch_f0(self, frame_len: int, lower: float = 75.0, upper: float = 600.0, method: str = 'pyin') -> PitchFrames:
        hop_len = frame_len//4
        if method == 'pyin':
            f0, flag, prob = self._calc_pitch_pyin(frame_len=frame_len, hop_len=hop_len, lower=lower, upper=upper)
        else:
            raise NotImplementedError('Only the "pyin" method is currently available')
        
        return PitchFrames(frames=f0, flag=flag, prob=prob, sr=self.sr, lower=lower, upper=upper, 
                           frame_len=frame_len, hop_len=hop_len, method=method)


class BaseFeature:
    def requires(self) -> Optional[Dict[str, type]]:
        return None


    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        return None


class FeaturePitchF0(BaseFeature):
    pitch_frames: PitchFrames = None


    def requires(self) -> Optional[Dict[str, type]]:
        return {'pitch_frames': PitchFrames}


    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        f_interp = interp1d(self.pitch_frames.ts, self.pitch_frames.frames, kind='linear')
        return f_interp(time)


class VoiceExtractor:
    """Extract voice features from an audio file.

    Currently, only the voice pitch as the fundamental frequency F0 can be extracted.
    The F0 is calculated using an autocorrelation function with a lower boundary of 75 Hz and an
    upper boudnary of 600 Hz. See the praat 
    `manual <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch___.html>`_ for details.

    """


    def __init__(self, features: Optional[Dict[str, BaseFeature]] = None):
        self.logger = logging.getLogger('mexca.audio.extraction.VoiceExtractor')

        if features is None:
            features = self._set_default_features()

        self.features = features

        self.logger.debug(ClassInitMessage())


    @staticmethod
    def _set_default_features():
        return {
            'pitch_f0': FeaturePitchF0()
        }


    def apply(self, filepath: str, time_step: float, skip_frames: int = 1) -> VoiceFeatures:
        """Extract voice features from an audio file.

        Parameters
        ----------
        filepath: str
            Path to the audio file.
        time_step: float
            The interval between time points at which features are extracted.
        skip_frames: int
            Only process every nth frame, starting at 0.

        Returns
        -------
        VoiceFeatures
            A data class object containing the extracted voice features.

        """
        self.logger.debug('Loading audio file')
        audio_signal = AudioSignal.from_file(filename=filepath)

        self.logger.debug('Extracting features with time step: %s', time_step)

        time = np.arange(audio_signal.ts.min(), audio_signal.ts.max(), time_step, dtype=np.float32)
        frame = np.array((time / time_step) * skip_frames, dtype=np.int32)
        
        pitch_frames = audio_signal.to_pitch_f0(frame_len=1024)

        requirements = [audio_signal, pitch_frames]
        requirements_types = [type(r) for r in requirements]

        extracted_features = VoiceFeatures(frame=frame.tolist(), time=time.tolist())
        extracted_features.add_attributes(self.features.keys())
        
        for key, feat in self.features.items():
            for attr, req in feat.requires().items():
                idx = requirements_types.index(req)
                feat.__setattr__(attr, requirements[idx])

            self.logger.debug('Extracting feature %s', key)
            extracted_features.add_feature(
                key,
                feat.apply(time)
            )

        return extracted_features


def cli():
    """Command line interface for extracting voice features.
    See `extract-voice -h` for details.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('-t', '--time-step', type=float, dest='time_step')
    parser.add_argument('--skip-frames', type=int, default=1, dest='skip_frames')

    args = parser.parse_args().__dict__

    extractor = VoiceExtractor()

    output = extractor.apply(args['filepath'], time_step=args['time_step'], skip_frames=args['skip_frames'])

    output.write_json(os.path.join(args['outdir'], os.path.splitext(os.path.basename(args['filepath']))[0] + '_voice_features.json'))


if __name__ == '__main__':
    cli()
    
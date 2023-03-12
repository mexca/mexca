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


class FormantFrames(BaseFrames):
    max_formants = 5
    lower = 50
    upper = 5450
    preemphasis_from =  50


    def __init__(self, frames: np.ndarray, sr: int, frame_len: int, hop_len: int, ) -> None:
        self.logger = logging.getLogger('mexca.audio.extraction.FormantFrames')
        super().__init__(frames, sr, frame_len, hop_len)
        self._calc_formants()
        self.logger.debug(ClassInitMessage())


    def _apply_window(self):
        window = get_window(('gauss', 1.0), self.frame_len)
        self.frames = np.multiply(self.frames, window)


    def _preemphasize(self):
        coef = math.exp(-2 * math.pi * self.preemphasis_from * (1/self.sr))
        self.frames = librosa.effects.preemphasis(self.frames, coef)


    def _calc_formants(self):
        self._preemphasize()
        self._apply_window()
        coefs = librosa.lpc(self.frames, order=self.max_formants*2)
        roots = np.array([np.roots(coef) for coef in coefs])
        mask = np.imag(roots) > 0
        roots[~mask] = np.nan
        ang_freq = np.arctan2(np.imag(roots), np.real(roots))
        self.formants = np.sort(ang_freq * (self.sr / (2 * math.pi)))
        self.logger.debug("%s", self.formants)


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
    

    def to_formants(self, frame_len: int) -> FormantFrames:
        hop_len = frame_len//4
        padding = [(0, 0) for _ in self.sig.shape]
        padding[-1] = (frame_len // 2, frame_len // 2)
        sig_pad = np.pad(self.sig, padding, mode='constant')
        frames = librosa.util.frame(sig_pad, frame_length=frame_len, hop_length=hop_len, axis=0)
        return FormantFrames(frames, self.sr, frame_len, hop_len)


class BasePulses:
    pulses: Optional[List] = None


    def __init__(self, audio_signal: AudioSignal, frames_obj: BaseFrames):
        self.logger = logging.getLogger('mexca.audio.extraction.BasePulses')
        self.audio_signal = audio_signal
        self.frames_obj = frames_obj
        self._sig = self.audio_signal.sig
        self._ts_sig = self.audio_signal.ts

        self._detect_pulses()
        self.logger.debug(ClassInitMessage())


    def _pad_audio_signal(self):
        padding = [(0, 0) for _ in self.audio_signal.sig.shape]
        padding[-1] = (self.frames_obj.frame_len // 2, self.frames_obj.frame_len // 2)
        self._sig = np.pad(self.audio_signal.sig, padding, mode=self.frames_obj.pad_mode)
        self._ts_sig = librosa.samples_to_time(np.arange(self._sig.shape[0]), sr=self.audio_signal.sr)


    def _framer(self, sig: np.ndarray) -> np.ndarray:
        return librosa.util.frame(
            sig,
            frame_length=self.frames_obj.frame_len,
            hop_length=self.frames_obj.hop_len,
            axis=0
        )


    def _frame_signal(self):
        self._sig_frames = self._framer(self._sig)
        self._ts_sig_frames = self._framer(self._ts_sig)
        self._ts_mid_sig_frames = np.apply_along_axis(lambda x: x.min() + (x.max() - x.min())/2, 1, self._ts_sig_frames)


    def _interpolate_frames(self):
        self._frames_interp_sig = np.interp(self._ts_sig, self.frames_obj.ts[self.frames_obj.flag], self.frames_obj.frames[self.frames_obj.flag])
        self._frames_interp_sig_frames = self._framer(self._frames_interp_sig)


    def _get_next_pulse(self,
        sig: np.ndarray,
        ts: np.ndarray,
        frames_interp: np.ndarray,
        start: float,
        stop: float,
        left: bool = True,
        pulses: List = []
    ):
        # If interval [start, stop] reaches end of frame, exit recurrence
        if (left and start <= ts.min()) or (not left and stop >= ts.max()) or np.isnan(start) or np.isnan(stop):
            return pulses
        
        # Get closest ts to boundaries start, stop
        start_idx = np.argmin(np.abs(ts - start))
        stop_idx = np.argmin(np.abs(ts - stop))
        interval = sig[start_idx:stop_idx]

        # Find max peak in interval [start, stop]
        peak_idx = np.nanargmax(interval)

        # Set new mid point to idx of max peak
        new_ts_mid = ts[start_idx:stop_idx][peak_idx]

        # Add pulse to output
        new_frames_interp_mid = frames_interp[start_idx:stop_idx][peak_idx]
        pulses.append((new_ts_mid, new_frames_interp_mid, interval[peak_idx]))

        # self.logger.debug('%s - %s - %s', start, stop, pulses)

        if left: # Move interval to left
            start = new_ts_mid - 1.25 * new_frames_interp_mid
            stop = new_ts_mid - 0.8 * new_frames_interp_mid
        else: # Move interval to right
            stop = new_ts_mid + 1.25 * new_frames_interp_mid
            start = new_ts_mid + 0.8 * new_frames_interp_mid

        # Find next pulse in new interval
        return self._get_next_pulse(sig, ts, frames_interp, start, stop, left, pulses)


class PitchPulses(BasePulses):
    def _detect_pulses_in_frame(self, frame_idx: int):
        t0_mid = 1/self.frames_obj.frames[frame_idx]
        ts_mid = self.frames_obj.ts[frame_idx]
        sig_frame = self._sig_frames[frame_idx, :]
        ts_sig_frame = self._ts_sig_frames[frame_idx, :]
        t0 = 1/self._frames_interp_sig_frames[frame_idx, :]

        pulses = []

        if np.all(np.isnan(t0)) or np.isnan(t0_mid):
            return pulses

        start = ts_mid - t0_mid/2
        stop = ts_mid + t0_mid/2

        self._get_next_pulse(sig_frame, ts_sig_frame, t0, start, stop, True, pulses)
        self._get_next_pulse(sig_frame, ts_sig_frame, t0, start, stop, False, pulses)

        return list(sorted(set(pulses)))


    def _detect_pulses(self):
        if self.frames_obj.center:
            self._pad_audio_signal()

        self._frame_signal()
        self._interpolate_frames()

        self.pulses = [self._detect_pulses_in_frame(i) for i in self.frames_obj.idx]


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


class BasePitchPulsesFeature(BaseFeature):
    pitch_pulses: PitchPulses = None


    def __init__(self, rel: bool = True, lower: float = 0.0001, upper: float = 0.02, max_period_ratio: float = 1.3):
        self.rel = rel
        self.lower = lower
        self.upper = upper
        self.max_period_ratio = max_period_ratio
        self._feature = None
        super().__init__()


    @property
    def feature(self):
        if self._feature is None:
            self._calc_feature()
            return self._feature
        return self._feature


    def requires(self) -> Optional[Dict[str, type]]:
        return {'pitch_pulses': PitchPulses}
    

    def _calc_period_length(self, pulses_idx: int) -> Tuple[List, np.ndarray]:
        # Calc period length as first order diff of pulse ts
        periods = np.diff(np.array([puls[0] for puls in self.pitch_pulses.pulses[pulses_idx]]))

        # Filter out too short and long periods
        mask = np.logical_and(periods > self.lower, periods < self.upper)

        # Split periods according to mask and remove masked periods
        periods = np.array_split(periods[mask], np.where(~mask)[0])

        return periods, mask
    

    def _get_amplitude(self, pulses_idx: int) -> Tuple[List, List]:
        # Get amplitudes
        amps = np.array([puls[2] for puls in self.pitch_pulses.pulses[pulses_idx]])[1:] # Skip first amplitude to align with periods

        # Calc period length and get mask for filtering amplitudes
        periods, mask = self._calc_period_length(pulses_idx)

        # Split periods according to mask and remove masked periods
        amps = np.array_split(amps[mask], np.where(~mask)[0])

        return amps, periods


    def _calc_feature(self):
        self._feature = np.array([self._calc_feature_frame(i) for i in range(len(self.pitch_pulses.pulses))])


    def _calc_feature_frame(self, idx: int) -> float:
        return np.nan
    

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        f_interp = interp1d(self.pitch_pulses.frames_obj.ts, self.feature, kind='linear')
        return f_interp(time)


class FeatureJitter(BasePitchPulsesFeature):
    def _calc_feature_frame(self, pulses_idx: int) -> float:
        if len(self.pitch_pulses.pulses[pulses_idx]) > 0:
            # Calc period length as first order diff of pulse ts
            periods, _ = self._calc_period_length(pulses_idx)

            # Calc avg of first order diff in period length
            # only consider period pairs where ratio is < max_period_ratio
            avg_period_diff = np.nanmean(np.array([np.mean(np.abs(np.diff(period)[
                                (period[:-1]/period[1:]) < self.max_period_ratio
                            ])) for period in periods if len(period) > 0]))

            if self.rel: # Relative to mean period length
                avg_period_len = np.nanmean(np.array([np.mean(period) for period in periods if len(period) > 0]))
                return avg_period_diff/avg_period_len
            return avg_period_diff     
        return np.nan


class FeatureShimmer(BasePitchPulsesFeature):
    def __init__(self, rel: bool = True, lower: float = 0.0001, upper: float = 0.02, max_period_ratio: float = 1.3, max_amp_factor: float = 1.6):
        self.max_amp_factor = max_amp_factor
        super().__init__(rel, lower, upper, max_period_ratio)


    def _calc_feature_frame(self, pulses_idx: int) -> float:
        if len(self.pitch_pulses.pulses[pulses_idx]) > 0:
            # Calc period length as first order diff of pulse ts
            amps, periods = self._get_amplitude(pulses_idx)

            # Calc avg of first order diff in period length
            # only consider period pairs where ratio is < max_period_ratio
            avg_amp_diff = np.nanmean(np.array([np.mean(np.abs(np.diff(amp)[
                                np.logical_and((period[:-1]/period[1:]) < self.max_period_ratio, (amp[:-1]/amp[1:]) < self.max_amp_factor)
                            ])) for amp, period in zip(amps, periods) if len(period) > 0]))

            if self.rel: # Relative to mean period length
                avg_amp = np.nanmean(np.array([np.mean(amp) for amp in amps if len(amp) > 0]))
                return avg_amp_diff/avg_amp
            return avg_amp_diff     
        return np.nan


class FeatureFormantFreq(BaseFeature):
    formants: FormantFrames = None


    def __init__(self, n_formant: int):
        self.n_formant = n_formant


    def requires(self) -> Optional[Dict[str, type]]:
        return {'formants': FormantFrames}
    

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        f_interp = interp1d(self.formants.ts, self.formants.formants[:, self.n_formant], kind='linear')
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
            'pitch_f0': FeaturePitchF0(),
            'jitter_rel': FeatureJitter(),
            'shimmer_rel': FeatureShimmer(),
            'f1_freq': FeatureFormantFreq(n_formant=0)
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
        pitch_pulses = PitchPulses(audio_signal=audio_signal, frames_obj=pitch_frames)
        formants = audio_signal.to_formants(frame_len=1024)

        requirements = [audio_signal, pitch_frames, pitch_pulses, formants]
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
    
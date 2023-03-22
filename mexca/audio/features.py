"""Compute audio signal properties to extract voice features.
"""

import logging
import math
from typing import List, Optional, Tuple, Union
import librosa
import numpy as np
from scipy.signal.windows import get_window
from mexca.utils import ClassInitMessage


class BaseSignal:
    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None

    def __init__(self, sig: np.ndarray, sr: float) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.BaseSignal")
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


class AudioSignal(BaseSignal):
    def __init__(self, sig: np.ndarray, sr: float, filename: str, mono: bool) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.AudioSignal")
        self.filename = filename
        self.mono = mono
        super().__init__(sig, sr)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_file(cls, filename: str, sr: Optional[int] = None, mono: bool = True):
        sig, nat_sr = librosa.load(path=filename, sr=sr, mono=mono)
        return cls(sig, nat_sr, filename, mono)


class BaseFrames:
    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.BaseFrames")
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
            self._ts = librosa.frames_to_time(
                self.idx, sr=self.sr, hop_length=self.hop_len
            )
        return self._ts

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
    ):
        if hop_len is None:
            hop_len = frame_len // 4
        sig = sig_obj.sig
        # Pad signal if centering
        if center:
            padding = [(0, 0) for _ in sig_obj.sig.shape]
            padding[-1] = (frame_len // 2, frame_len // 2)
            sig = np.pad(sig, padding, mode=pad_mode)

        frames = librosa.util.frame(
            sig, frame_length=frame_len, hop_length=hop_len, axis=0
        )

        return cls(frames, sig_obj.sr, frame_len, hop_len, center, pad_mode)


class PitchFrames(BaseFrames):
    def __init__(
        self,
        frames: np.ndarray,
        flag: np.ndarray,
        prob: np.ndarray,
        sr: int,
        lower: float,
        upper: float,
        frame_len: int,
        hop_len: int,
        method: str,
        center: bool = True,
        pad_mode: str = "constant",
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.PitchFrames")
        self.flag = flag
        self.probs = prob
        self.lower = lower
        self.upper = upper
        self.method = method
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
        lower: float = 75.0,
        upper: float = 600.0,
        method: str = "pyin",
    ):
        if method == "pyin":
            pitch_f0, flag, prob = librosa.pyin(
                sig_obj.sig,
                fmin=lower,
                fmax=upper,
                sr=sig_obj.sr,
                frame_length=frame_len,
                hop_length=hop_len,
                center=center,
                pad_mode=pad_mode,
            )
        else:
            raise NotImplementedError('Only the "pyin" method is currently available')

        return cls(
            frames=pitch_f0,
            flag=flag,
            prob=prob,
            sr=sig_obj.sr,
            lower=lower,
            upper=upper,
            frame_len=frame_len,
            hop_len=hop_len,
            method=method,
        )


class SpecFrames(BaseFrames):
    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.SpecFrames")
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
        window: Union[str, float, Tuple] = "hamming",
    ):
        spec_frames = librosa.stft(
            sig_obj.sig,
            n_fft=frame_len,
            hop_length=hop_len,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return cls(
            np.swapaxes(spec_frames, 0, 1),
            sig_obj.sr,
            frame_len,
            hop_len,
            center,
            pad_mode,
        )


class FormantFrames(BaseFrames):
    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: float = 50.0,
        window: Union[str, float, Tuple] = "hamming",
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.FormantFrames")
        self.max_formants = max_formants
        self.lower = lower
        self.upper = upper
        self.preemphasis_from = preemphasis_from
        self.window = window
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @classmethod
    def from_frames(
        cls,
        sig_frames_obj: BaseFrames,
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: float = 50.0,
        window: Union[str, float, Tuple] = "hamming",
    ):
        frames = sig_frames_obj.frames

        if preemphasis_from is not None:
            pre_coef = math.exp(
                -2 * math.pi * preemphasis_from * (1 / sig_frames_obj.sr)
            )
            frames = librosa.effects.preemphasis(sig_frames_obj.frames, coef=pre_coef)
        if window is not None:
            win = get_window(window, sig_frames_obj.frame_len, fftbins=False)
            frames = frames * win

        # Calc linear predictive coefficients
        coefs = librosa.lpc(frames, order=max_formants * 2)
        # Transform LPCs to formants
        formants = [
            cls._calc_formants(coef, sig_frames_obj.sr, lower, upper) for coef in coefs
        ]

        return cls(
            formants,
            sig_frames_obj.sr,
            sig_frames_obj.frame_len,
            sig_frames_obj.hop_len,
            sig_frames_obj.center,
            sig_frames_obj.pad_mode,
            lower,
            upper,
            preemphasis_from,
            window,
        )

    @staticmethod
    def _calc_formants(
        coefs: np.ndarray, sr: int, lower: float = 50, upper: float = 5450
    ) -> List:
        # Function to compute complex norm
        def complex_norm(x):
            return np.sqrt(np.abs(np.real(x) ** 2) + np.abs(np.imag(x) ** 2))

        nf_pi = sr / (2 * math.pi)  # sr/2 = Nyquist freq
        # Find roots of linear coefficients
        roots = np.roots(coefs)
        # Select roots with positive imag part
        mask = np.imag(roots) > 0
        roots = roots[mask]
        # Calc angular frequency
        ang_freq = np.abs(np.arctan2(np.imag(roots), np.real(roots)))
        # Calc formant centre freq
        formant_freqs = ang_freq * nf_pi
        # Calc formant bandwidth
        formant_bws = -np.log(np.apply_along_axis(complex_norm, 0, roots)) * nf_pi
        # Select formants within boundaries
        in_bounds = np.logical_and(formant_freqs > lower, formant_freqs < upper)
        formants_sorted = sorted(
            list(zip(formant_freqs[in_bounds], formant_bws[in_bounds]))
        )
        return formants_sorted


class PitchHarmonicsFrames(BaseFrames):
    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
        n_harmonics: int = 100,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.PitchHarmonicsFrames")
        self.n_harmonics = n_harmonics
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_and_pitch(
        cls, spec_frames_obj: SpecFrames, pitch_frames_obj: PitchFrames, n_harmonics: int = 12
    ):
        freqs = librosa.fft_frequencies(sr=spec_frames_obj.sr, n_fft=spec_frames_obj.frame_len)

        harmonics = librosa.f0_harmonics(
            np.abs(spec_frames_obj.frames),
            freqs=freqs,
            f0=pitch_frames_obj.frames,
            harmonics=np.arange(n_harmonics) + 1,  # Shift one up
            axis=-1,
        )

        return cls(
            harmonics,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            n_harmonics,
        )


class PitchPulseFrames(BaseFrames):
    def __init__(
        self,
        frames: List[Tuple],
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.PitchPulseFrames")
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @classmethod
    def from_signal_and_pitch(cls, sig_obj: BaseSignal, pitch_frames_obj: PitchFrames):
        # Access to padded signal required so we transform it here again! Could go into separate private method perhaps
        padding = [(0, 0) for _ in sig_obj.sig.shape]
        padding[-1] = (pitch_frames_obj.frame_len // 2, pitch_frames_obj.frame_len // 2)
        sig_padded = np.pad(sig_obj.sig, padding, mode=pitch_frames_obj.pad_mode)
        # Create ts for padded signal
        sig_padded_ts = librosa.samples_to_time(
            np.arange(sig_padded.shape[0]), sr=sig_obj.sr
        )

        # Frame padded signal
        sig_frames_obj = BaseFrames.from_signal(
            BaseSignal(sig_padded, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Frame ts of padded signal
        sig_ts_frames_obj = BaseFrames.from_signal(
            BaseSignal(sig_padded_ts, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Interpolate pitch F0 at padded signal ts
        interp_f0 = np.interp(
            sig_padded_ts,
            pitch_frames_obj.ts[pitch_frames_obj.flag],
            pitch_frames_obj.frames[pitch_frames_obj.flag],
        )

        # Frame F0 interpolated signal
        pitch_interp_frames_obj = BaseFrames.from_signal(
            BaseSignal(interp_f0, sig_obj.sr),
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            center=False,
        )

        # Detect pulses in each frame; objects are passed instead of arrays bcs some attributes are needed
        pulses = [
            cls._detect_pulses_in_frame(
                i,
                sig_frames_obj,
                sig_ts_frames_obj,
                pitch_frames_obj,
                pitch_interp_frames_obj,
            )
            for i in pitch_frames_obj.idx
        ]

        return cls(
            pulses,
            pitch_frames_obj.sr,
            pitch_frames_obj.frame_len,
            pitch_frames_obj.hop_len,
            pitch_frames_obj.center,
            pitch_frames_obj.pad_mode,
        )

    @classmethod
    def _get_next_pulse(
        cls,
        sig: np.ndarray,
        ts: np.ndarray,
        t0_interp: np.ndarray,
        start: float,
        stop: float,
        left: bool = True,
        pulses: Optional[List] = None,
    ):
        # Init pulses as list if first iter of recurrence and default
        if pulses is None:
            pulses = []

        # If interval [start, stop] reaches end of frame, exit recurrence
        if (
            (left and start <= ts.min())
            or (not left and stop >= ts.max())
            or np.isnan(start)
            or np.isnan(stop)
        ):
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
        new_t0_interp_mid = t0_interp[start_idx:stop_idx][peak_idx]
        pulses.append((new_ts_mid, new_t0_interp_mid, interval[peak_idx]))

        # self.logger.debug('%s - %s - %s', start, stop, pulses)

        if left:  # Move interval to left
            start = new_ts_mid - 1.25 * new_t0_interp_mid
            stop = new_ts_mid - 0.8 * new_t0_interp_mid
        else:  # Move interval to right
            stop = new_ts_mid + 1.25 * new_t0_interp_mid
            start = new_ts_mid + 0.8 * new_t0_interp_mid

        # Find next pulse in new interval
        return cls._get_next_pulse(sig, ts, t0_interp, start, stop, left, pulses)

    @classmethod
    def _detect_pulses_in_frame(
        cls,
        frame_idx: int,
        sig_frames_obj: BaseFrames,
        sig_ts_frames_obj: BaseFrames,
        pitch_obj: PitchFrames,
        pitch_interp_obj: BaseFrames,
    ) -> List[Tuple]:
        # Get period of frame
        t0_mid = 1 / pitch_obj.frames[frame_idx]
        # Get ts of frame
        ts_mid = pitch_obj.ts[frame_idx]
        # Get frame signal
        sig_frame = sig_frames_obj.frames[frame_idx, :]
        # Get ts of frame signal
        ts_sig_frame = sig_ts_frames_obj.frames[frame_idx, :]
        # Get interpolated period of frame
        t0 = 1 / pitch_interp_obj.frames[frame_idx, :]

        pulses = []

        # Return empty list if frame is unvoiced (no F0)
        if np.all(np.isnan(t0)) or np.isnan(t0_mid):
            return pulses

        # Set start interval
        start = ts_mid - t0_mid / 2
        stop = ts_mid + t0_mid / 2

        # Get pulses to the left
        cls._get_next_pulse(sig_frame, ts_sig_frame, t0, start, stop, True, pulses)

        # Get pulses to the right
        cls._get_next_pulse(sig_frame, ts_sig_frame, t0, start, stop, False, pulses)

        return list(sorted(set(pulses)))

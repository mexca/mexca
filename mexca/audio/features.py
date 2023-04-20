"""Compute audio signal properties to extract voice features.

This module contains classes and methods to compute and store properties of audio signals
that can be used to extract voice features.

There are two main types of classes: *Signal* (inherits from `BaseSignal`) and
*Frames* (inherits from `BaseFrames`). Signals contain data about an entire
signal (e.g., the audio signal itself) whereas Frames contain transformed and
aggregated data about overlapping slices of the signal.

"""

import logging
import math
from copy import copy
from typing import List, Optional, Tuple, Union
import librosa
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal.windows import get_window
from sklearn.linear_model import LinearRegression
from mexca.utils import ClassInitMessage


class BaseSignal:
    """Store a signal.

    Parameters
    ----------
    sig: numpy.ndarray
        Signal.
    sr: int
        Sampling rate.
    """

    _ts: Optional[np.ndarray] = None
    _idx: Optional[np.ndarray] = None

    def __init__(self, sig: np.ndarray, sr: int) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.BaseSignal")
        self.sig = sig
        self.sr = sr
        self.logger.debug(ClassInitMessage())

    @property
    def idx(self) -> np.ndarray:
        """Sample indices (read-only)."""
        if self._idx is None:
            self._idx = np.arange(self.sig.shape[0])
        return self._idx

    @property
    def ts(self) -> np.ndarray:
        """Sample timestamps (read-only)."""
        if self._ts is None:
            self._ts = librosa.samples_to_time(self.idx, sr=self.sr)
        return self._ts


class AudioSignal(BaseSignal):
    """Load and store an audio signal.

    Parameters
    ----------
    sig: numpy.ndarray
        Audio signal.
    sr: int
        Sampling rate.
    mono: bool, default=True
        Whether the signal has been converted to mono or not.
    filename: str, optional
        Name of the audio file associated with the signal.
    """

    def __init__(
        self,
        sig: np.ndarray,
        sr: int,
        mono: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.AudioSignal")
        self.filename = filename
        self.mono = mono
        super().__init__(sig, sr)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_file(cls, filename: str, sr: Optional[float] = None, mono: bool = True):
        """Load a signal from an audio file.

        Parameters
        ----------
        filename: str
            Name of the audio file.
            File types must be supported by ``soundfile`` or ``audiofile``.
            See :func:`librosa.load`.
        sr: float, optional, default=None
            Sampling rate. If `None`, is detected from the file, otherwise the signal is resampled.
        mono: bool, default=True
            Whether to convert the signal to mono.
        """
        sig, nat_sr = librosa.load(path=filename, sr=sr, mono=mono)
        return cls(sig, nat_sr, mono, filename)


class FormantAudioSignal(AudioSignal):
    def __init__(
        self,
        sig: np.ndarray,
        sr: int,
        mono: bool,
        filename: Optional[str],
        preemphasis_from: Optional[float],
    ):
        self.preemphasis_from = preemphasis_from
        super().__init__(sig, sr, mono, filename)

    @staticmethod
    def _calc_preemphasis_coef(preemphasis_from: float, sr: float) -> float:
        return math.exp(-2 * math.pi * preemphasis_from * (1 / sr))

    @classmethod
    def from_audio_signal(
        cls, audio_sig_obj: AudioSignal, preemphasis_from: Optional[float] = 50.0
    ):
        sig = audio_sig_obj.sig

        if preemphasis_from is not None:
            pre_coef = cls._calc_preemphasis_coef(preemphasis_from, audio_sig_obj.sr)
            sig = librosa.effects.preemphasis(sig, coef=pre_coef)

        return cls(
            sig,
            audio_sig_obj.sr,
            audio_sig_obj.mono,
            audio_sig_obj.filename,
            preemphasis_from,
        )

    @staticmethod
    def _preemphasize(sig: np.ndarray, sr: float, preemphasis_from: float):
        pre_coef = math.exp(-2 * math.pi * preemphasis_from * (1 / sr))
        return librosa.effects.preemphasis(sig, coef=pre_coef)


class BaseFrames:
    """Create and store signal frames.

    A frame is an (overlapping, padded) slice of a signal for which higher-order
    features can be computed.

    Parameters
    ----------
    frames: numpy.ndarray
        Signal frames. The first dimension should be the number of frames.
    sr: int
        Sampling rate.
    frame_len: int
        Number of samples per frame.
    hop_len: int
        Number of samples between frame starting points.
    center: bool, default=True
        Whether the signal has been centered and padded before framing.
    pad_mode: str, default='constant'
        How the signal has been padded before framing. See :func:`numpy.pad`.
        Uses the default value 0 for `'constant'` padding.

    See Also
    --------
    librosa.util.frame

    """

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
        """Frame indices (read-only)."""
        if self._idx is None:
            self._idx = np.arange(self.frames.shape[0])
        return self._idx

    @property
    def ts(self) -> np.ndarray:
        """Frame timestamps (read-only)."""
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
        """Create frames from a signal.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        frame_len: int
            Number of samples per frame.
        hop_len: int, optional, default=None
            Number of samples between frame starting points. If `None`, uses `frame_len // 4`.
        center: bool, default=True
            Whether to center the frames and apply padding.
        pad_mode: str, default='constant'
            How the signal is padded before framing. See :func:`numpy.pad`.
            Uses the default value 0 for `'constant'` padding. Ignored if `center=False`.
        """
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
    """Estimate and store pitch frames.

    Estimate and store the voice pitch measured as the fundamental frequency F0 in Hz.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice pitch frames in Hz with shape (num_frames,).
    flag: numpy.ndarray
        Boolean flags indicating which frames are voiced with shape (num_frames,).
    prob: numpy.ndarray
        Probabilities for frames being voiced with shape (num_frames,).
    lower: float
        Lower limit used for pitch estimation (in Hz).
    upper: float
        Upper limit used for pitch estimation (in Hz).
    method: str
        Method used for estimating voice pitch.

    See Also
    --------
    librosa.pyin
    librosa.yin

    """

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
        """Estimate the voice pitch frames from a signal.

        Currently, voice pitch can only be extracted with the *pYIN* method.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        frame_len: int
            Number of samples per frame.
        hop_len: int, optional, default=None
            Number of samples between frame starting points. If `None`, uses `frame_len // 4`.
        center: bool, default=True
            Whether to center the frames and apply padding.
        pad_mode: str, default='constant'
            How the signal is padded before framing. See :func:`numpy.pad`.
            Uses the default value 0 for `'constant'` padding. Ignored if `center=False`.
        lower: float, default = 75.0
            Lower limit for pitch estimation (in Hz).
        upper: float, default = 600.0
            Upper limit for pitch estimation (in Hz).
        method: str, default = 'pyin'
            Method for estimating voice pitch. Only `'pyin'` is currently available.

        Raises
        ------
        NotImplementedError
            If a method other than `'pyin'` is given.

        """
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
    """Create and store spectrogram frames.

    Computes a spectrogram of a signal using the short-time Fourier transform (STFT).

    Parameters
    ----------
    frames: numpy.ndarray
        Spectrogram frames.
    window: str
        The window that was applied before the STFT.

    Notes
    -----
    Frames contain complex arrays `x` where ``np.abs(x)`` is the magnitude and
    ``np.angle(x)`` is the phase of the signal for different frequency bins.

    See Also
    --------
    librosa.stft

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.SpecFrames")
        self.window = window
        self._freqs = None
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @property
    def freqs(self):
        if self._freqs is None:
            self._freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.frame_len)
        return self._freqs

    @classmethod
    def from_signal(
        cls,
        sig_obj: BaseSignal,
        frame_len: int,
        hop_len: Optional[int] = None,
        center: bool = True,
        pad_mode: str = "constant",
        window: Union[str, float, Tuple] = "hann",
    ):
        """Transform a signal into spectrogram frames.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        frame_len: int
            Number of samples per frame.
        hop_len: int, optional, default=None
            Number of samples between frame starting points. If `None`, uses `frame_len // 4`.
        center: bool, default=True
            Whether to center the frames and apply padding.
        pad_mode: str, default='constant'
            How the signal is padded before framing. See :func:`numpy.pad`.
            Uses the default value 0 for `'constant'` padding. Ignored if `center=False`.
        window: str
            The window that is applied before the STFT.
        """
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
            window,
            frame_len,
            hop_len,
            center,
            pad_mode,
        )


class FormantFrames(BaseFrames):
    """Estimate and store formant frames.

    Parameters
    ----------
    frames: list
        Formant frames. Each frame contains a list of tuples for each formant, where the first item
        is the central frequency and the second the bandwidth.
    max_formants: int, default=5
        The maximum number of formants that were extracted.
    lower: float, default=50.0
        Lower limit for formant frequencies (in Hz).
    upper: float, default=5450.0
        Upper limit for formant frequencies (in Hz).
    preemphasis_from: float, default=50.0
        Starting value for the applied preemphasis function.
    window: str
        Window function that was applied before formant estimation.

    Notes
    -----
    Estimate formants of the signal in each frame:

    1. Apply a preemphasis function with the coefficient
       ``math.exp(-2 * math.pi * preemphasis_from * (1 / sr))``
       to the signal.
    2. Apply a window function to the signal. By default, the same Gaussian window as in
       Praat is used: ``(np.exp(-48.0 * (n - ((N + 1)/2)**2 / (N + 1)**2) - np.exp(-12.0)) / (1.0 - np.exp(-12.0))``,
       where `N` is the length of the window and `n` the index of each sample.
    3. Calculate linear predictive coefficients using :func:`librosa.lpc`
       with order ``2 * max_formants``.
    4. Find the roots of the coefficients.
    5. Compute the formant central frequencies as
       ``np.abs(np.arctan2(np.imag(roots), np.real(roots))) * sr / (2 * math.pi)``.
    6. Compute the formant bandwidth as
       ``np.sqrt(np.abs(np.real(roots) ** 2) + np.abs(np.imag(roots) ** 2)) * sr / (2 * math.pi)``.
    7. Filter out formants outside the lower and upper limits.

    """

    def __init__(
        self,
        frames: List,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool = True,
        pad_mode: str = "constant",
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: Optional[float] = 50.0,
        window: Optional[Union[str, float, Tuple]] = "praat_gaussian",
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

    @staticmethod
    def _praat_gauss_window(frame_len: int):
        # This is the Gaussian window that is used in Praat for formant estimation
        # See: https://github.com/YannickJadoul/Parselmouth/blob/master/praat/fon/Sound_to_Formant.cpp
        sample_idx = np.arange(frame_len) + 1
        idx_mid = 0.5 * (frame_len + 1)
        edge = np.exp(-12.0)
        return (
            np.exp(-48.0 * (sample_idx - idx_mid) ** 2 / (frame_len) ** 2) - edge
        ) / (1.0 - edge)

    @classmethod
    def from_frames(
        cls,
        sig_frames_obj: BaseFrames,
        max_formants: int = 5,
        lower: float = 50.0,
        upper: float = 5450.0,
        preemphasis_from: Optional[float] = 50.0,
        window: Optional[Union[str, float, Tuple]] = "praat_gaussian",
    ):
        """Extract formants from signal frames.

        Parameters
        ----------
        sig_frames_obj: BaseFrames
            Signal frames object.
        max_formants: int, default=5
            The maximum number of formants that are extracted.
        lower: float, default=50.0
            Lower limit for formant frequencies (in Hz).
        upper: float, default=5450.0
            Upper limit for formant frequencies (in Hz).
        preemphasis_from: float, default=50.0
            Starting value for the preemphasis function (in Hz).
        window: str
            Window function that is applied before formant estimation.

        """
        frames = sig_frames_obj.frames

        if preemphasis_from is not None:
            pre_coef = math.exp(
                -2 * math.pi * preemphasis_from * (1 / sig_frames_obj.sr)
            )
            frames = librosa.effects.preemphasis(sig_frames_obj.frames, coef=pre_coef)
        if window is not None:
            if window == "praat_gaussian":
                win = cls._praat_gauss_window(sig_frames_obj.frame_len)
            else:
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
            max_formants,
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

    def select_formant_attr(self, formant_idx: int, attr_idx: int) -> np.ndarray:
        return np.array(
            [
                f[formant_idx][attr_idx] if len(f) > formant_idx else np.nan
                for f in self.frames
            ]
        )


class PitchHarmonicsFrames(BaseFrames):
    """Estimate and store voice pitch harmonics.

    Compute the energy of the signal at harmonics (`nF0` for any integer n) of
    the fundamental frequency.

    Parameters
    ----------
    frames: numpy.ndarray
        Harmonics frames with the shape (num_frames, n_harmonics)
    n_harmonics: int, default=100
        Number of estimated harmonics.

    See Also
    --------
    librosa.f0_harmonics

    """

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
    def from_spec_and_pitch_frames(
        cls,
        spec_frames_obj: SpecFrames,
        pitch_frames_obj: PitchFrames,
        n_harmonics: int = 100,
    ):
        """Estimate voice pitch harmonics from spectrogram frames and voice pitch frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        pitch_frames_obj: PitchFrames
            Pitch frames object.
        n_harmonics: int, default=100
            Number of harmonics to estimate.

        """

        # harmonics = librosa.f0_harmonics(
        #     np.abs(spec_frames_obj.frames),
        #     freqs=freqs,
        #     f0=pitch_frames_obj.frames,
        #     harmonics=np.arange(n_harmonics) + 1,  # Shift one up
        #     axis=-1,
        # )

        harmonics = cls._calc_f0_harmonics(
            spec_frames_obj.frames,
            spec_frames_obj.freqs,
            pitch_frames_obj.frames,
            n_harmonics,
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

    @staticmethod
    def _calc_f0_harmonics(
        spec_frames: np.ndarray,
        freqs: np.ndarray,
        f0_frames: np.ndarray,
        n_harmonics: int,
    ) -> np.ndarray:
        # Adapted from librosa.f0_harmonics, see:
        # https://librosa.org/doc/latest/generated/librosa.f0_harmonics.html#librosa.f0_harmonics
        is_valid = np.isfinite(freqs)

        def mag_interp_fun(spec_frames, f0_harmonic_freqs):
            interp = interp1d(
                freqs[is_valid],
                spec_frames[is_valid],
                axis=0,
                copy=False,
                assume_sorted=False,
                bounds_error=False,
                fill_value=0,
            )
            return interp(f0_harmonic_freqs)

        xfunc = np.vectorize(mag_interp_fun, signature="(f),(h)->(h)")
        harmonics_frames = xfunc(
            np.abs(spec_frames),
            np.multiply.outer(f0_frames, np.arange(n_harmonics) + 1),  # Shift one up
        )

        return harmonics_frames


class FormantAmplitudeFrames(BaseFrames):
    """Estimate and store formant amplitudes.

    Parameters
    ----------
    frames: np.ndarray
        Formant amplitude frames of shape (num_frames, max_formants) in dB.
    lower: float
        Lower boundary for peak amplitude search interval.
    upper: float
        Upper boundary for peak amplitude search interval.
    rel_f0: bool
        Whether the amplitude is relative to the fundamental frequency amplitude.

    Notes
    -----
    Estimate the formant amplitude as the maximum amplitude of harmonics of the
    fundamental frequency within an interval ``[lower*f, upper*f]`` where `f` is the
    central frequency of the formant in each frame. If ``rel=True``, divide the amplitude by the amplitude of
    the fundamental frequency.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        upper: float,
        rel_f0: bool,
    ):
        self.lower = lower
        self.upper = upper
        self.rel_f0 = rel_f0
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)

    @property
    def idx(self) -> np.ndarray:
        if self._idx is None:
            self._idx = np.arange(len(self.frames))
        return self._idx

    @classmethod
    def from_formant_harmonics_and_pitch_frames(  # pylint: disable=too-many-locals
        cls,
        formant_frames_obj: FormantFrames,
        harmonics_frames_obj: PitchHarmonicsFrames,
        pitch_frames_obj: PitchFrames,
        lower: float = 0.8,
        upper: float = 1.2,
        rel_f0: bool = True,
    ):
        """Estimate formant amplitudes from formant, pitch harmonics, and pitch frames.

        Parameters
        ----------
        formant_frames_obj: FormantFrames
            Formant frames object.
        harmonics_frames_obj: PitchHarmonicsFrames
            Pitch harmonics frames object.
        pitch_frames_obj: PitchFrames
            Pitch frames object.
        lower: float, optional, default=0.8
            Lower boundary for peak amplitude search interval.
        upper: float, optional, default=1.2
            Upper boundary for peak amplitude search interval.
        rel_f0: bool, optional, default=True
            Whether the amplitude is divided by the fundamental frequency amplitude.
        """
        amp_frames = []

        for i in range(formant_frames_obj.max_formants):
            freqs = formant_frames_obj.select_formant_attr(i, 0)
            harmonic_freqs = (
                pitch_frames_obj.frames[:, None]
                * (np.arange(harmonics_frames_obj.n_harmonics) + 1)[None, :]
            )
            f0_amp = harmonics_frames_obj.frames[:, 0]
            freqs_lower = lower * freqs
            freqs_upper = upper * freqs
            freq_in_bounds = np.logical_and(
                harmonic_freqs > freqs_lower[:, None],
                harmonic_freqs < freqs_upper[:, None],
            )
            harmonics_amp = copy(harmonics_frames_obj.frames)
            harmonics_amp[~freq_in_bounds] = np.nan
            # Set all-nan frames to nan (otherwise np.nanmax throws warning)
            harmonic_peaks = np.zeros(harmonics_amp.shape[0:1])
            harmonics_amp_all_na = np.all(np.isnan(harmonics_amp), axis=1)
            harmonic_peaks[harmonics_amp_all_na] = np.nan
            harmonic_peaks[~harmonics_amp_all_na] = np.nanmax(
                harmonics_amp[~harmonics_amp_all_na], axis=1
            )
            harmonic_peaks_db = librosa.amplitude_to_db(harmonic_peaks)

            if rel_f0:
                harmonic_peaks_db = harmonic_peaks_db - librosa.amplitude_to_db(f0_amp)

            amp_frames.append(harmonic_peaks_db)

        return cls(
            np.array(amp_frames).T,
            formant_frames_obj.sr,
            formant_frames_obj.frame_len,
            formant_frames_obj.hop_len,
            formant_frames_obj.center,
            formant_frames_obj.pad_mode,
            lower,
            upper,
            rel_f0,
        )


class PitchPulseFrames(BaseFrames):
    """Extract and store glottal pulse frames.

    Glottal pulses are peaks in the signal corresponding to the fundamental frequency F0.

    Parameters
    ----------
    frames: list
        Pulse frames. Each frame contains a list of pulses or an empty list if no pulses are detected.
        Pulses are stored as tuples (pulse timestamp, T0, amplitude).

    Notes
    -----
    Extract glottal pulses with these steps:

    1. Interpolate the fundamental frequency at the timestamps of the framed (padded) signal.
    2. Start at the mid point `m` of each frame and create an interval [start, stop],
       where ``start=m-T0/2`` and
       ``stop=m+T0/2`` and `T0` is the fundamental period (1/F0).
    3. Detect pulses in the interval by:
        a. Find the maximum amplitude in an interval within the frame.
        b. Compute the fundamental period `T0_new` at the timestamp of the maximum `m_new`.
    4. Shift the interval recursively to the right or left until the edges of the frame are reached:
        a. When shifting to the left, set ``start_new=m_new-1.25*T0_new``
           and ``stop_new=m_new-0.8*T0_new``.
        b. When shifting to the right, set ``start_new=m_new+0.8*T0_new``
           and ``stop_new=m_new+1.25*T0_new``.
    5. Filter out duplicate pulses.

    """

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
    def from_signal_and_pitch_frames(
        cls, sig_obj: BaseSignal, pitch_frames_obj: PitchFrames
    ):
        """Extract glottal pulse frames from a signal and voice pitch frames.

        Parameters
        ----------
        sig_obj: BaseSignal
            Signal object.
        pitch_frames_obj: PitchFrames
            Voice pitch frames object.

        """
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
            or any(np.isnan((start, stop)))
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


class PitchPeriodFrames(BaseFrames):
    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        upper: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.PitchPeriodFrames")
        self.lower = lower
        self.upper = upper
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @staticmethod
    def _calc_period_length(
        pulses: List[Tuple], lower: float, upper: float
    ) -> Tuple[List, np.ndarray]:
        # Calc period length as first order diff of pulse ts
        periods = np.diff(np.array([puls[0] for puls in pulses]))

        # Filter out too short and long periods
        mask = np.logical_and(periods > lower, periods < upper)

        # Split periods according to mask and remove masked periods
        periods = np.array_split(periods[mask], np.where(~mask)[0])

        return periods, mask

    @staticmethod
    def _check_ratio(x_arr: np.ndarray, threshold: float) -> np.ndarray:
        valid = np.logical_and(np.isfinite(x_arr[1:]), x_arr[1:] > 0)
        valid[valid] = x_arr[:-1][valid] / x_arr[1:][valid] < threshold
        return valid


class JitterFrames(PitchPeriodFrames):
    """Extract and store voice jitter frames.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice jitter frames of shape (num_frames,).
    rel: bool
        Whether the voice jitter is relative to the average period length.
    lower: float
        Lower limit for periods between glottal pulses.
    upper: float
        Upper limit for periods between glottal pulses.
    max_period_ratio: float
        Maximum ratio between consecutive periods used for jitter extraction.

    Notes
    -----
    Compute jitter as the average absolute difference between consecutive fundamental periods with a ratio
    below `max_period_ratio` for each frame. If ``rel=True``, jitter is divided by the average fundamental period
    of each frame. Fundamental periods are calculated as the first-order temporal difference between consecutive
    glottal pulses.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.JitterFrames")
        self.rel = rel
        self.max_period_ratio = max_period_ratio
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode, lower, upper)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_pitch_pulse_frames(
        cls,
        pitch_pulse_frames_obj: PitchPulseFrames,
        rel: bool = True,
        lower: float = 0.0001,
        upper: float = 0.02,
        max_period_ratio: float = 1.3,
    ):
        """Extract voice jitter frames from glottal pulse frames.

        Parameters
        ----------
        pitch_pulse_frames_obj: PitchPulseFrames
            Glottal pulse frames object.
        rel: bool, optional, default=True
            Divide jitter by the average pitch period.
        lower: float, optional, default=0.0001
            Lower limit for periods between glottal pulses.
        upper: float, optional, default=0.02
            Upper limit for periods between glottal pulses.
        max_period_ratio: float, optional, default=1.3
            Maximum ratio between consecutive periods for jitter extraction.
        """
        jitter_frames = np.array(
            [
                cls._calc_jitter_frame(pulses, rel, lower, upper, max_period_ratio)
                for pulses in pitch_pulse_frames_obj.frames
            ]
        )

        return cls(
            jitter_frames,
            pitch_pulse_frames_obj.sr,
            pitch_pulse_frames_obj.frame_len,
            pitch_pulse_frames_obj.hop_len,
            pitch_pulse_frames_obj.center,
            pitch_pulse_frames_obj.pad_mode,
            rel,
            lower,
            upper,
            max_period_ratio,
        )

    @classmethod
    def _calc_jitter_frame(
        cls,
        pulses: List[Tuple],
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
    ):
        if len(pulses) == 0:
            return np.nan

        # Calc period length as first order diff of pulse ts
        periods, _ = cls._calc_period_length(pulses, lower, upper)

        if len(periods) == 0 or all(len(period) <= 1 for period in periods):
            return np.nan

        # Calc avg of first order diff in period length
        # only consider period pairs where ratio is < max_period_ratio
        period_diff = [
            np.abs(np.diff(period)[cls._check_ratio(period, max_period_ratio)])
            for period in periods
            if len(period) > 1
        ]

        if len(period_diff) == 0 or all(len(period) == 0 for period in period_diff):
            return np.nan

        avg_period_diff = np.nanmean(
            np.array([np.mean(period) for period in period_diff])
        )

        if rel:  # Relative to mean period length
            avg_period_len = np.nanmean(
                np.array([np.mean(period) for period in periods if len(period) > 1])
            )
            return avg_period_diff / avg_period_len

        return avg_period_diff


class ShimmerFrames(PitchPeriodFrames):
    """Extract and store voice shimmer frames.

    Parameters
    ----------
    frames: numpy.ndarray
        Voice shimmer frames of shape (num_frames,).
    rel: bool
        Whether the voice shimmer is relative to the average period length.
    lower: float
        Lower limit for periods between glottal pulses.
    upper: float
        Upper limit for periods between glottal pulses.
    max_period_ratio: float
        Maximum ratio between consecutive periods used for shimmer extraction.
    max_amp_factor: float
        Maximum ratio between consecutive amplitudes used for shimmer extraction.

    Notes
    -----
    Compute shimmer as the average absolute difference between consecutive pitch amplitudes with a
    fundamental period ratio below `max_period_ratio` and amplitude ratio below `max_amp_factor`
    for each frame. If ``rel=True``, shimmer is divided by the average amplitude
    of each frame. Fundamental periods are calculated as the first-order temporal difference
    between consecutive glottal pulses. Amplitudes are signal amplitudes at the glottal pulses.
    """

    def __init__(
        self,
        frames: List[Tuple],
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
        max_amp_factor: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.ShimmerFrames")
        self.rel = rel
        self.max_period_ratio = max_period_ratio
        self.max_amp_factor = max_amp_factor
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode, lower, upper)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_pitch_pulse_frames(
        cls,
        pitch_pulse_frames_obj: PitchPulseFrames,
        rel: bool = True,
        lower: float = 0.0001,
        upper: float = 0.02,
        max_period_ratio: float = 1.3,
        max_amp_factor: float = 1.6,
    ):
        """Extract voice shimmer frames from glottal pulse frames.

        Parameters
        ----------
        pitch_pulse_frames_obj: PitchPulseFrames
            Glottal pulse frames object.
        rel: bool, optional, default=True
            Divide shimmer by the average pulse amplitude.
        lower: float, optional, default=0.0001
            Lower limit for periods between glottal pulses.
        upper: float, optional, default=0.02
            Upper limit for periods between glottal pulses.
        max_period_ratio: float, optional, default=1.3
            Maximum ratio between consecutive periods for shimmer extraction.
        max_amp_factor: float, optional, default=1.6
            Maximum ratio between consecutive amplitudes used for shimmer extraction.
        """
        shimmer_frames = np.array(
            [
                cls._calc_shimmer_frame(
                    pulses, rel, lower, upper, max_period_ratio, max_amp_factor
                )
                for pulses in pitch_pulse_frames_obj.frames
            ]
        )

        return cls(
            shimmer_frames,
            pitch_pulse_frames_obj.sr,
            pitch_pulse_frames_obj.frame_len,
            pitch_pulse_frames_obj.hop_len,
            pitch_pulse_frames_obj.center,
            pitch_pulse_frames_obj.pad_mode,
            rel,
            lower,
            upper,
            max_period_ratio,
            max_amp_factor,
        )

    @classmethod
    def _calc_shimmer_frame(
        cls,
        pulses: List[Tuple],
        rel: bool,
        lower: float,
        upper: float,
        max_period_ratio: float,
        max_amp_factor: float,
    ) -> float:
        if len(pulses) == 0:
            return np.nan

        # Calc period length as first order diff of pulse ts
        periods, mask = cls._calc_period_length(pulses, lower, upper)
        amps = cls._get_amplitude(pulses, mask)

        if (
            len(periods) == 0
            or len(amps) == 0
            or all(len(period) <= 1 for period in periods)
        ):
            return np.nan

        # Calc avg of first order diff in amplitude
        # only consider period pairs where period ratio is < max_period_ratio and
        # where amplitude ratio is < max_amp_factor
        amp_diff = [
            np.abs(
                np.diff(amp)[
                    np.logical_and(
                        cls._check_ratio(period, max_period_ratio),
                        cls._check_ratio(amp, max_amp_factor),
                    )
                ]
            )
            for amp, period in zip(amps, periods)
            if len(period) > 1 and len(amp) > 1
        ]

        if len(amp_diff) == 0 or all(len(amp) == 0 for amp in amp_diff):
            return np.nan

        avg_amp_diff = np.nanmean(np.array([np.mean(amp) for amp in amp_diff]))

        if rel:  # Relative to mean amplitude
            avg_amp = np.nanmean(
                np.array([np.mean(amp) for amp in amps if len(amp) > 1])
            )
            return avg_amp_diff / avg_amp

        return avg_amp_diff

    @staticmethod
    def _get_amplitude(pulses: List[Tuple], mask: np.ndarray) -> List:
        # Get amplitudes
        amps = np.array([puls[2] for puls in pulses])[
            1:
        ]  # Skip first amplitude to align with periods

        # Split periods according to mask and remove masked periods
        amps = np.array_split(amps[mask], np.where(~mask)[0])

        return amps


class HnrFrames(BaseFrames):
    """Estimate and store harmonics-to-noise ratios (HNRs).

    Parameters
    ----------
    frames: numpy.ndarray
        HNR frames in dB with shape (num_frames,).
    lower: float
        Lower fundamental frequency limit for choosing pitch candidates.
    rel_silence_threshold: float
        Relative threshold for treating signal frames as silent.

    Notes
    -----
    Estimate the HNR for each signal frame with ``np.max(np.abs(frames), axis=1) > rel_silence_threshold*np.max(np.abs(frames))`` by:

    1. Compute the autocorrelation function (ACF) using the short-term Fourier transform (STFT).
    2. Find the lags of peaks in the ACF excluding the zero-th lag.
    3. Filter out peaks that correspond to pitch candidates below `lower` and above the Nyquist frequency.
    4. Compute the harmonic component `R0` as the highest of the remaining peaks divided by the ACF at lag zero.
    5. Compute the HNR as `R0/(1-R0)` and convert to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        rel_silence_threshold,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.HnrFrames")
        self.lower = lower
        self.rel_silence_threshold = rel_silence_threshold
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_frames(
        cls,
        sig_frames_obj: BaseFrames,
        lower: float = 75.0,
        rel_silence_threshold: float = 0.1,
    ):
        """Estimate the HNR from signal frames.

        Parameters
        ----------
        sig_frames_obj: BaseFrames
            Signal frames object.
        lower: float, default = 75.0
            Lower fundamental frequency limit for choosing pitch candidates.
        rel_silence_threshold: float, default = 0.1
            Relative threshold for treating signal frames as silent.

        """
        auto_cor = librosa.autocorrelate(sig_frames_obj.frames)
        harmonic_strength = np.apply_along_axis(
            cls._find_max_peak, 1, auto_cor[:, 1:], sr=sig_frames_obj.sr, lower=lower
        )
        harmonic_comp = harmonic_strength / auto_cor[:, 0]
        hnr = harmonic_comp / (1 - harmonic_comp)
        silence_mask = np.max(
            np.abs(sig_frames_obj.frames), axis=1
        ) > rel_silence_threshold * np.max(np.abs(sig_frames_obj.frames))
        hnr[np.logical_or(~silence_mask, hnr <= 0)] = np.nan
        hnr_db = librosa.power_to_db(hnr)  # HNR is on power scale
        return cls(
            hnr_db,
            sig_frames_obj.sr,
            sig_frames_obj.frame_len,
            sig_frames_obj.hop_len,
            sig_frames_obj.center,
            sig_frames_obj.pad_mode,
            lower,
            rel_silence_threshold,
        )

    @staticmethod
    def _find_max_peak(auto_cor: np.ndarray, sr: int, lower: float) -> float:
        if np.all(np.isnan(auto_cor)):
            return np.nan

        auto_cor_peak_lags = find_peaks(auto_cor)[0]
        auto_cor_peaks = auto_cor[auto_cor_peak_lags]
        auto_cor_peak_periods = 1 / auto_cor_peak_lags * sr
        auto_cor_peaks_voiced = auto_cor_peaks[
            np.logical_and(
                auto_cor_peak_periods > lower, auto_cor_peak_periods < sr / 2
            )
        ]

        if len(auto_cor_peaks_voiced) == 0:
            return np.nan

        auto_cor_max_peak_lag = np.argmax(auto_cor_peaks_voiced)

        return auto_cor_peaks_voiced[auto_cor_max_peak_lag]


class AlphaRatioFrames(BaseFrames):
    """Calculate and store spectogram alpha ratios.

    Parameters
    ----------
    frames: numpy.ndarray
        Alpha ratio frames in dB with shape (num_frames,).
    lower_band: tuple
        Boundaries of the lower frequency band (start, end) in Hz.
    upper_band: tuple
        Boundaries of the upper frequency band (start, end) in Hz.

    Notes
    -----
    Calculate the alpha ratio by dividing the energy (sum of magnitude) in the lower frequency band
    by the energy in the upper frequency band. The ratio is then converted to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower_band: Tuple[float],
        upper_band: Tuple[float],
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.AlphaRatioFrames")
        self.lower_band = lower_band
        self.upper_band = upper_band
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        lower_band: Tuple = (50.0, 1000.0),
        upper_band: Tuple = (1000.0, 5000.0),
    ):
        """Calculate the alpha ratio from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        lower_band: tuple, default=(50.0, 1000.0)
            Boundaries of the lower frequency band (start, end) in Hz.
        upper_band: tuple, default=(1000.0, 5000.0)
            Boundaries of the upper frequency band (start, end) in Hz.

        """
        lower_band_bins = np.logical_and(
            spec_frames_obj.freqs > lower_band[0],
            spec_frames_obj.freqs <= lower_band[1],
        )
        lower_band_energy = np.nansum(
            np.abs(spec_frames_obj.frames[:, lower_band_bins]), axis=1
        )
        upper_band_bins = np.logical_and(
            spec_frames_obj.freqs > upper_band[0],
            spec_frames_obj.freqs <= upper_band[1],
        )
        upper_band_energy = np.nansum(
            np.abs(spec_frames_obj.frames[:, upper_band_bins]), axis=1
        )
        alpha_ratio_frames = np.zeros(lower_band_energy.shape)

        upper_band_energy_is_valid = np.logical_and(
            np.isfinite(upper_band_energy), upper_band_energy != 0
        )

        alpha_ratio_frames[~upper_band_energy_is_valid] = np.nan
        alpha_ratio_frames[upper_band_energy_is_valid] = (
            lower_band_energy[upper_band_energy_is_valid]
            / upper_band_energy[upper_band_energy_is_valid]
        )

        alpha_ratio_frames_db = 20.0 * np.log10(alpha_ratio_frames)

        return cls(
            alpha_ratio_frames_db,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            lower_band,
            upper_band,
        )


class HammarIndexFrames(BaseFrames):
    """Calculate and store the spectogram Hammarberg index.

    Parameters
    ----------
    frames: numpy.ndarray
        Hammarberg index frames in dB with shape (num_frames,).
    pivot_point: float
        Point separating the lower and upper frequency regions in Hz.
    upper: float
        Upper limit for the upper frequency region in Hz.

    Notes
    -----
    Calculate the Hammarberg index by dividing the peak magnitude in the spectrogram region below `pivot_point`
    by the peak magnitude in region between `pivot_point` and `upper`. The ratio is then converted to dB.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        pivot_point: float,
        upper: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.HammarIndexFrames")
        self.pivot_point = pivot_point
        self.upper = upper
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        pivot_point: float = 2000.0,
        upper: float = 5000.0,
    ):
        """Calculate the Hammarberg index from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        pivot_point: float, default=2000.0
            Point separating the lower and upper frequency regions in Hz.
        upper: float, default=5000.0
            Upper limit for the upper frequency region in Hz.

        """
        lower_band = np.abs(
            spec_frames_obj.frames[:, spec_frames_obj.freqs <= pivot_point]
        )
        upper_band_freqs = np.logical_and(
            spec_frames_obj.freqs > pivot_point, spec_frames_obj.freqs <= upper
        )
        upper_band = np.abs(spec_frames_obj.frames[:, upper_band_freqs])

        hammar_index_frames = np.zeros(lower_band.shape[0])

        upper_band_is_valid = np.logical_and(
            np.any(np.isfinite(upper_band), axis=1), np.all(upper_band > 0, axis=1)
        )

        hammar_index_frames[~upper_band_is_valid] = np.nan
        hammar_index_frames[upper_band_is_valid] = np.nanmax(
            lower_band[upper_band_is_valid, :], axis=1
        ) / np.nanmax(upper_band[upper_band_is_valid, :], axis=1)

        hammar_index_frames_db = librosa.amplitude_to_db(hammar_index_frames)

        return cls(
            hammar_index_frames_db,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            pivot_point,
            upper,
        )


class SpectralSlopeFrames(BaseFrames):
    """Estimate and store spectral slopes.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectral slope frames with shape (num_frames, num_bands).
    bands: tuple
        Frequency bands in Hz for which slopes were estimated.

    Notes
    -----
    Estimate spectral slopes by fitting linear models to frequency bands predicting power in dB from frequency in Hz.
    Fits separate models for each frame and band.

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        bands: Tuple[Tuple[float]],
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.HammarIndexFrames")
        self.bands = bands
        super().__init__(frames, sr, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        bands: Tuple[Tuple[float]] = ((0.0, 500.0), (500.0, 1500.0)),
    ):
        """Estimate spectral slopes from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        bands: tuple, default=((0.0, 500.0), (500.0, 1500.0))
            Frequency bands in Hz for which slopes are estimated.

        """
        spectral_slopes = np.zeros(shape=(spec_frames_obj.idx.shape[0], len(bands)))

        for i, band in enumerate(bands):
            band_freqs_mask = np.logical_and(
                spec_frames_obj.freqs > band[0], spec_frames_obj.freqs <= band[1]
            )
            band_power = np.abs(spec_frames_obj.frames[:, band_freqs_mask])
            band_freqs = spec_frames_obj.freqs[band_freqs_mask]
            spectral_slopes[:, i] = np.apply_along_axis(
                cls._calc_spectral_slope, 1, band_power, band_freqs=band_freqs
            ).squeeze()

        return cls(
            spectral_slopes,
            spec_frames_obj.sr,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            bands,
        )

    @staticmethod
    def _calc_spectral_slope(
        band_power: np.ndarray, band_freqs: np.ndarray
    ) -> np.ndarray:
        band_power_is_valid = np.logical_and(np.isfinite(band_power), band_power > 0)

        if np.all(~band_power_is_valid):
            return np.nan

        band_freqs_finite = band_freqs[band_power_is_valid]
        band_power_finite_db = librosa.amplitude_to_db(band_power[band_power_is_valid])

        linear_model = LinearRegression()
        linear_model.fit(band_freqs_finite.reshape(-1, 1), band_power_finite_db)
        return linear_model.coef_


class MelSpecFrames(SpecFrames):
    """Calculate and store Mel spectrograms.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectrogram frames on the Mel power scale with shape (num_frames, n_mels).
    n_mels: int
        Number of Mel filters.
    lower: float
        Lower frequency boundary in Hz.
    upper: float
        Upper frequency boundary in Hz.

    See Also
    --------
    librosa.feature.melspectrogram

    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        n_mels: int,
        lower: float,
        upper: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.MelSpecFrames")
        self.n_mels = n_mels
        self.lower = lower
        self.upper = upper
        super().__init__(frames, sr, window, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_frames(
        cls,
        spec_frames_obj: SpecFrames,
        n_mels: int = 26,
        lower: float = 20.0,
        upper: float = 8000.0,
    ):
        """Calculate Mel spectrograms from spectrogram frames.

        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        n_mels: int, default=26
            Number of Mel filters.
        lower: float, default=20.0
            Lower frequency boundary in Hz.
        upper: float, default=8000.0
            Upper frequency boundary in Hz.

        """
        mel_spec_frames = librosa.feature.melspectrogram(
            S=np.abs(spec_frames_obj.frames.T) ** 2,  # requires power spectrum
            sr=spec_frames_obj.sr,
            n_fft=spec_frames_obj.frame_len,
            hop_length=spec_frames_obj.hop_len,
            window=spec_frames_obj.window,
            center=spec_frames_obj.center,
            pad_mode=spec_frames_obj.pad_mode,
            n_mels=n_mels,
            fmin=lower,
            fmax=upper,
        )

        return cls(
            mel_spec_frames.T,  # outputs power spectrum
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            n_mels,
            lower,
            upper,
        )


class MfccFrames(MelSpecFrames):
    """Estimate and store Mel frequency cepstral coefficients (MFCCs).

    Parameters
    ----------
    frames: numpy.ndarray
        MFCC frames with shape (num_frames, n_mfcc).
    n_mfcc: int
        Number of coeffcients that were estimated per frame.
    lifter: float
        Cepstral liftering coefficient. Must be >= 0. If zero, no liftering is applied.


    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        n_mels: int,
        lower: float,
        upper: float,
        n_mfcc: int,
        lifter: float,
    ):
        self.logger = logging.getLogger("mexca.audio.extraction.MfccFrames")
        self.n_mfcc = n_mfcc
        self.lifter = lifter
        super().__init__(
            frames,
            sr,
            window,
            frame_len,
            hop_len,
            center,
            pad_mode,
            n_mels,
            lower,
            upper,
        )
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_mel_spec_frames(
        cls, mel_spec_frames_obj: MelSpecFrames, n_mfcc: int = 4, lifter: float = 22.0
    ):
        """Estimate MFCCs from Mel spectogram frames.

        Parameters
        ----------
        mel_spec_frames_obj: MelSpecFrames
            Mel spectrogram frames object.
        n_mfcc: int, default=4
            Number of coeffcients that were estimated per frame.
        lifter: float, default=22.0
            Cepstral liftering coefficient. Must be >= 0. If zero, no liftering is applied.

        See Also
        --------
        librosa.feature.mfcc

        """
        mfcc_frames = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec_frames_obj.frames.T),  # dB on power spectrum
            sr=mel_spec_frames_obj.sr,
            n_mfcc=n_mfcc,
            lifter=lifter,
        )

        return cls(
            mfcc_frames.T,
            mel_spec_frames_obj.sr,
            mel_spec_frames_obj.window,
            mel_spec_frames_obj.frame_len,
            mel_spec_frames_obj.hop_len,
            mel_spec_frames_obj.center,
            mel_spec_frames_obj.pad_mode,
            mel_spec_frames_obj.n_mels,
            mel_spec_frames_obj.lower,
            mel_spec_frames_obj.upper,
            n_mfcc,
            lifter,
        )


class SpectralFluxFrames(SpecFrames):
    """Calculate and store spectral flux.

    Parameters
    ----------
    frames: numpy.ndarray
        Spectral flux frames with shape (num_frames-1,).
    lower: float
        Lower limit for frequency bins.
    upper: float
        Upper limit for frequency bins

    Notes
    -----
    Compute the spectral flux as:

    1. Compute the normalized magnitudes of the frame spectra by dividing the magnitude
       at each frequency bin by the sum of all frequency bins.
    2. Compute the first-order difference of normalized magnitudes for each frequency bin within [`lower`, `upper`) across frames.
    3. Sum up the squared differences for each frame.

    Due to the first-order difference, the object has a frame less than the
    spectrogram from which it has been computed.


    """

    def __init__(
        self,
        frames: np.ndarray,
        sr: int,
        window: str,
        frame_len: int,
        hop_len: int,
        center: bool,
        pad_mode: str,
        lower: float,
        upper: float,
    ) -> None:
        self.logger = logging.getLogger("mexca.audio.extraction.SpectralFluxFrames")
        self.lower = lower
        self.upper = upper
        super().__init__(frames, sr, window, frame_len, hop_len, center, pad_mode)
        self.logger.debug(ClassInitMessage())

    @classmethod
    def from_spec_frames(
        cls, spec_frames_obj: SpecFrames, lower: float = 0.0, upper: float = 5000.0
    ):
        """Calculate the spectral flux from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.
        lower: float, default=0.0
            Lower limit for frequency bins.
        upper: float, default=5000.0
            Upper limit for frequency bins

        """
        spec_freq_mask = np.logical_and(
            spec_frames_obj.freqs >= lower, spec_frames_obj.freqs < upper
        )
        spec_mag = np.abs(spec_frames_obj.frames)
        spec_norm = np.sum(spec_mag, axis=1)
        spec_diff = np.diff(spec_mag[:, spec_freq_mask] / spec_norm[:, None], axis=0)
        spec_flux_frames = np.sum(spec_diff**2, axis=1)

        return cls(
            spec_flux_frames,
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
            lower,
            upper,
        )


class RmsEnergyFrames(SpecFrames):
    """Calculate and store the root mean squared (RMS) energy.

    Parameters
    ---------
    frames: numpy.ndarray
        RMS energy frames in dB with shape (num_frames,).

    """

    @classmethod
    def from_spec_frames(cls, spec_frames_obj: SpecFrames):
        """Calculate the RMS energy from spectrogram frames.

        Parameters
        ----------
        spec_frames_obj: SpecFrames
            Spectrogram frames object.

        """
        rms_frames = librosa.amplitude_to_db(
            librosa.feature.rms(  # to dB
                S=np.abs(spec_frames_obj.frames).T,
                frame_length=spec_frames_obj.frame_len,
                hop_length=spec_frames_obj.hop_len,
                center=spec_frames_obj.center,
                pad_mode=spec_frames_obj.pad_mode,
            )
        )

        return cls(
            rms_frames.squeeze(),
            spec_frames_obj.sr,
            spec_frames_obj.window,
            spec_frames_obj.frame_len,
            spec_frames_obj.hop_len,
            spec_frames_obj.center,
            spec_frames_obj.pad_mode,
        )

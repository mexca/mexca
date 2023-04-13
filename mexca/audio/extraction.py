"""Extract voice features from an audio file.

Construct a dictionary with keys as feature names and values as feature objects. The dictionary can
be used to extract the specified features with the :class:`VoiceExtractor`. Feature objects require
lower-level voice signal properties, which are defined in the :func:`requires` method of feach 
feature class. The :class:`VoiceExtractor` class computes the properties and supplies them to 
the feature objects.

"""

import argparse
import logging
import os
from typing import Dict, Optional, Union
import numpy as np
from scipy.interpolate import interp1d
from mexca.audio.features import (AlphaRatioFrames, AudioSignal, BaseFrames, FormantAmplitudeFrames, FormantAudioSignal,
                                  FormantFrames, HammarIndexFrames, HnrFrames, JitterFrames, MelSpecFrames, MfccFrames,
                                  PitchFrames, PitchHarmonicsFrames, PitchPulseFrames, RmsEnergyFrames, ShimmerFrames,
                                  SpecFrames, SpectralFluxFrames, SpectralSlopeFrames)
from mexca.data import VoiceFeatures, VoiceFeaturesConfig
from mexca.utils import ClassInitMessage, optional_str


class BaseFeature:
    """Base class for features.

    Can be used to create custom voice feature extraction classes.
    """

    def requires(self) -> Optional[Dict[str, type]]:
        """Specify objects required for feature extraction.

        This method can be overwritten to return a dictionary with keys as the names of objects
        required for computing features and values the types of these objects. The :class:`VoiceExtractor`
        object will look for objects with the specified types and add them as attributes to the feature
        class with the names of the dictionary keys.

        Returns
        -------
        dict
            Dictionary where keys are the names and values the types of required objects.

        """
        return None

    def _get_interp_fun(self, ts: np.ndarray, feature: np.ndarray) -> np.ndarray:
        return interp1d(ts, feature, kind="linear", bounds_error=False)

    def apply(self, time: np.ndarray) -> np.ndarray:
        """Extract features at time points by linear interpolation.

        Parameters
        ----------
        time: numpy.ndarray
            Time points.

        Returns
        -------
        numpy.ndarray
            Feature values interpolated at time points.

        """
        return time


class FeaturePitchF0(BaseFeature):
    """Extract voice pitch as the fundamental frequency F0 in Hz."""

    pitch_frames: Optional[PitchFrames] = None

    def requires(self) -> Optional[Dict[str, PitchFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `pitch_frames`.

        """
        return {"pitch_frames": PitchFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.pitch_frames.ts, self.pitch_frames.frames)(
            time
        )


class FeatureJitter(BaseFeature):
    """Extract local jitter relative to the fundamental frequency."""

    jitter_frames: Optional[JitterFrames] = None

    def requires(self) -> Optional[Dict[str, JitterFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `jitter_frames`.

        """
        return {"jitter_frames": JitterFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.jitter_frames.ts, self.jitter_frames.frames)(
            time
        )


class FeatureShimmer(BaseFeature):
    """Extract local shimmer relative to the fundamental frequency."""

    shimmer_frames: Optional[ShimmerFrames] = None

    def requires(self) -> Optional[Dict[str, ShimmerFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `shimmer_frames`.

        """
        return {"shimmer_frames": ShimmerFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.shimmer_frames.ts, self.shimmer_frames.frames)(
            time
        )


class FeatureHnr(BaseFeature):
    """Extract the harmonicity-to-noise ratio in dB."""

    hnr_frames: Optional[HnrFrames] = None

    def requires(self) -> Optional[Dict[str, HnrFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `hnr_frames`.

        """
        return {"hnr_frames": HnrFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.hnr_frames.ts, self.hnr_frames.frames)(time)


class FeatureFormantFreq(BaseFeature):
    """Extract formant central frequency in Hz.

    Parameters
    ----------
    n_formant: int
        Index of the formant (starting at 0).

    """

    formant_frames: Optional[FormantFrames] = None

    def __init__(self, n_formant: int):
        self.n_formant = n_formant

    def requires(self) -> Optional[Dict[str, FormantFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `formant_frames`.

        """
        return {"formant_frames": FormantFrames}

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_freqs = self.formant_frames.select_formant_attr(self.n_formant, 0)
        return self._get_interp_fun(self.formant_frames.ts, formants_freqs)(time)


class FeatureFormantBandwidth(FeatureFormantFreq):
    """Extract formant frequency bandwidth in Hz.

    Parameters
    ----------
    n_formant: int
        Index of the formant (starting at 0).

    """

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_bws = self.formant_frames.select_formant_attr(self.n_formant, 1)
        return self._get_interp_fun(self.formant_frames.ts, formants_bws)(time)


class FeatureFormantAmplitude(BaseFeature):
    """Extract formant amplitude relative to F0 harmonic amplitude.

    Parameters
    ----------
    n_formant: int
        Index of the formant (starting at 0).

    """

    formant_amp_frames: Optional[FormantAmplitudeFrames] = None

    def __init__(self, n_formant: int):
        self.n_formant = n_formant

    def requires(self) -> Optional[Dict[str, FormantAmplitudeFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `formant_amp_frames`.

        """
        return {"formant_amp_frames": FormantAmplitudeFrames}

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_amps = self.formant_amp_frames.frames
        return self._get_interp_fun(
            self.formant_amp_frames.ts, formants_amps[:, self.n_formant]
        )(time)


class FeatureAlphaRatio(BaseFeature):
    """Extract the alpha ratio in dB."""
    alpha_ratio_frames: Optional[AlphaRatioFrames] = None

    def requires(self) -> Optional[Dict[str, AlphaRatioFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `alpha_ratio_frames`.

        """
        return {"alpha_ratio_frames": AlphaRatioFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.alpha_ratio_frames.ts, self.alpha_ratio_frames.frames
        )(time)


class FeatureHammarIndex(BaseFeature):
    """Extract the Hammarberg index in dB."""
    hammar_index_frames: Optional[HammarIndexFrames] = None

    def requires(self) -> Optional[Dict[str, HammarIndexFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `hammar_index_frames`.

        """
        return {"hammar_index_frames": HammarIndexFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.hammar_index_frames.ts, self.hammar_index_frames.frames
        )(time)


class FeatureSpectralSlope(BaseFeature):
    """Extract spectral slopes for frequency bands.
    
    Parameters
    ----------
    lower, upper: float
        Lower and upper boundary of the frequency band for which to extract the spectral slope.
        A band with these boundaries must exist in the required `spectral_slope_frames` object.

    """
    spectral_slope_frames: Optional[SpectralSlopeFrames] = None

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def requires(self) -> Optional[Dict[str, type]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `spectral_slope_frames`.

        """
        return {"spectral_slope_frames": SpectralSlopeFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        slope_idx = self.spectral_slope_frames.bands.index((self.lower, self.upper))
        return self._get_interp_fun(
            self.spectral_slope_frames.ts,
            self.spectral_slope_frames.frames[:, slope_idx],
        )(time)


class FeatureHarmonicDifference(BaseFeature):
    """Extract the difference between pitch harmonic and/or formant amplitudes in dB.

    Parameters
    ----------
    x_idx, y_idx: int, default=0
        Index of the first/second amplitude.
    x_type, y_type: str, default='h'
        Type of the first/second amplitude. Must be either `'h'` for pitch harmonic or `'f'` for formant.

    Raises
    ------
    ValueError
        If `x_type` or `y_type` is not `'h'` or `'f'`.    
    
    """
    formant_amp_frames: Optional[FormantAmplitudeFrames] = None
    pitch_harmonics_frames: Optional[PitchHarmonicsFrames] = None

    def __init__(
        self, x_idx: int = 0, x_type: str = "h", y_idx: int = 1, y_type: str = "h"
    ):
        self.x_idx = x_idx
        self.x_type = x_type
        self.y_idx = y_idx
        self.y_type = y_type

    def requires(
        self,
    ) -> Optional[Dict[str, Union[FormantAmplitudeFrames, PitchHarmonicsFrames]]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with keys `formant_amp_frames` and `pitch_harmonics_frames`.

        """
        return {
            "formant_amp_frames": FormantAmplitudeFrames,
            "pitch_harmonics_frames": PitchHarmonicsFrames,
        }

    def _get_harmonic_or_formant(self, which: str = "x") -> np.ndarray:
        if getattr(self, which + "_type") == "h":
            var = 20 * np.log10(
                self.pitch_harmonics_frames.frames[:, getattr(self, which + "_idx")]
            )
        elif getattr(self, which + "_type") == "f":
            # Formant amplitude is already on dB scale
            var = self.formant_amp_frames.frames[:, getattr(self, which + "_idx")]
            # Multiply by F0 on dB scale
            if self.formant_amp_frames.rel_f0:
                var = var + 20 * np.log10(self.pitch_harmonics_frames.frames[:, 0])
        else:
            raise ValueError(
                f"'{which}_type' must be either 'h' (pitch harmonic) or 'f' (formant)"
            )

        return var

    def apply(self, time: np.ndarray) -> np.ndarray:
        x_var = self._get_harmonic_or_formant(which="x")
        y_var = self._get_harmonic_or_formant(which="y")

        ratio = x_var - y_var  # ratio on log scale

        return self._get_interp_fun(self.formant_amp_frames.ts, ratio)(time)


class FeatureMfcc(BaseFeature):
    """Extract Mel frequency cepstral coefficients (MFCCs).

    Parameters
    ----------
    n_mfcc: int, default=0
        Index of the MFCC to be extracted.

    """
    mfcc_frames: Optional[MfccFrames] = None

    def __init__(self, n_mfcc: int = 0) -> None:
        self.n_mfcc = n_mfcc

    def requires(self) -> Optional[Dict[str, MfccFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `mfcc_frames`.

        """
        return {"mfcc_frames": MfccFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.mfcc_frames.ts, self.mfcc_frames.frames[:, self.n_mfcc]
        )(time)


class FeatureSpectralFlux(BaseFeature):
    """Extract spectral flux.
    """
    spec_flux_frames: Optional[SpectralFluxFrames] = None

    def requires(self) -> Optional[Dict[str, SpectralFluxFrames]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `spectral_flux_frames`.

        """
        return {"spec_flux_frames": SpectralFluxFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.spec_flux_frames.ts, self.spec_flux_frames.frames
        )(time)


class FeatureRmsEnergy(BaseFeature):
    """Extract the root mean squared energy in dB.
    """
    rms_frames: Optional[RmsEnergyFrames] = None

    def requires(self) -> Optional[Dict[str, type]]:
        """Specify objects required for feature extraction.

        Returns
        -------
        dict
            Dictionary with key `rms_frames`.

        """
        return {"rms_frames": RmsEnergyFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.rms_frames.ts, self.rms_frames.frames)(time)


class VoiceExtractor:
    """Extract voice features from an audio file.

    For default features, see the :ref:`Output <voice_features_output>` section.

    Parameters
    ----------
    features: dict, optional, default=None
        Dictionary with keys as feature names and values as feature extraction objects. If `None`,
        default features are extracted.
    config: VoiceFeaturesConfig, optional, default=None
        Voice feature extraction configuration object. If `None`, uses :class:`VoiceFeaturesConfig`'s default configuration.

    """

    def __init__(self, features: Optional[Dict[str, BaseFeature]] = None, config: Optional[VoiceFeaturesConfig] = None):
        self.logger = logging.getLogger("mexca.audio.extraction.VoiceExtractor")

        if features is None:
            features = self._set_default_features()

        if config is None:
            config = VoiceFeaturesConfig()

        self._check_features(features)

        self.features = features
        self.config = config

        self.logger.debug(ClassInitMessage())

    @staticmethod
    def _check_features(features: dict):
        for key, item in features.items():
            if not isinstance(key, str):
                raise TypeError(f'Feature name {key} is not a string')
            if not isinstance(item, BaseFeature):
                raise TypeError(f'Feature object {item} with name {key} is not a subclass of "mexca.audio.features.BaseFeature"')

    @staticmethod
    def _set_default_features() -> Dict[str, BaseFeature]:
        return {
            "pitch_f0_hz": FeaturePitchF0(),
            "jitter_local_rel_f0": FeatureJitter(),
            "shimmer_local_rel_f0": FeatureShimmer(),
            "hnr_db": FeatureHnr(),
            "f1_freq_hz": FeatureFormantFreq(n_formant=0),
            "f1_bandwidth_hz": FeatureFormantBandwidth(n_formant=0),
            "f1_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=0),
            "f2_freq_hz": FeatureFormantFreq(n_formant=1),
            "f2_bandwidth_hz": FeatureFormantBandwidth(n_formant=1),
            "f2_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=1),
            "f3_freq_hz": FeatureFormantFreq(n_formant=2),
            "f3_bandwidth_hz": FeatureFormantBandwidth(n_formant=2),
            "f3_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=2),
            "alpha_ratio_db": FeatureAlphaRatio(),
            "hammar_index_db": FeatureHammarIndex(),
            "spectral_slope_0_500": FeatureSpectralSlope(lower=0, upper=500),
            "spectral_slope_500_1500": FeatureSpectralSlope(lower=500, upper=1500),
            "h1_h2_diff_db": FeatureHarmonicDifference(),
            "h1_f3_diff_db": FeatureHarmonicDifference(y_idx=2, y_type="f"),
            "mfcc_1": FeatureMfcc(),
            "mfcc_2": FeatureMfcc(n_mfcc=1),
            "mfcc_3": FeatureMfcc(n_mfcc=2),
            "mfcc_4": FeatureMfcc(n_mfcc=3),
            "spectral_flux": FeatureSpectralFlux(),
            "rms_db": FeatureRmsEnergy(),
        }

    def apply(  # pylint: disable=too-many-locals
        self, filepath: str, time_step: float, skip_frames: int = 1
    ) -> VoiceFeatures:
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
        self.logger.debug("Loading audio file")
        audio_signal = AudioSignal.from_file(filename=filepath)

        self.logger.debug("Extracting features with time step: %s", time_step)

        time = np.arange(
            audio_signal.ts.min(), audio_signal.ts.max(), time_step, dtype=np.float32
        )
        frame = np.array((time / time_step) * skip_frames, dtype=np.int32)

        sig_frames = BaseFrames.from_signal(
            audio_signal, frame_len=self.config.frame_len, hop_len=self.config.hop_len, center=self.config.center, pad_mode=self.config.pad_mode
        )

        spec_frames = SpecFrames.from_signal(
            audio_signal, frame_len=self.config.frame_len, hop_len=self.config.hop_len, center=self.config.center, pad_mode=self.config.pad_mode, window=self.config.spec_window
        )
        pitch_frames = PitchFrames.from_signal(
            audio_signal, frame_len=self.config.frame_len, hop_len=self.config.hop_len, center=self.config.center, pad_mode=self.config.pad_mode, lower=self.config.pitch_lower_freq, upper=self.config.pitch_upper_freq, method=self.config.pitch_method
        )
        pulses_frames = PitchPulseFrames.from_signal_and_pitch_frames(
            audio_signal, pitch_frames
        )
        jitter_frames = JitterFrames.from_pitch_pulse_frames(pulses_frames, rel=self.config.jitter_rel, lower=self.config.pitch_pulse_lower_period, upper=self.config.pitch_pulse_upper_period, max_period_ratio=self.config.pitch_pulse_max_period_ratio)
        shimmer_frames = ShimmerFrames.from_pitch_pulse_frames(pulses_frames, rel=self.config.shimmer_rel, lower=self.config.pitch_pulse_lower_period, upper=self.config.pitch_pulse_upper_period, max_period_ratio=self.config.pitch_pulse_max_period_ratio, max_amp_factor=self.config.pitch_pulse_max_amp_factor)
        hnr_frames = HnrFrames.from_frames(sig_frames, lower=self.config.hnr_lower_freq, rel_silence_threshold=self.config.hnr_rel_silence_threshold)

        formant_signal = FormantAudioSignal.from_audio_signal(
            audio_signal, preemphasis_from=self.config.formants_signal_preemphasis_from
        )
        formant_sig_frames = BaseFrames.from_signal(
            formant_signal, frame_len=self.config.frame_len, hop_len=self.config.hop_len, center=self.config.center, pad_mode=self.config.pad_mode
        )
        formant_frames = FormantFrames.from_frames(
            formant_sig_frames, max_formants=self.config.formants_max, lower=self.config.formants_lower_freq, upper=self.config.formants_upper_freq, preemphasis_from=None, window=self.config.formants_window
        )
        pitch_harmonics = PitchHarmonicsFrames.from_spec_and_pitch_frames(
            spec_frames, pitch_frames, n_harmonics=self.config.pitch_n_harmonics
        )
        formant_amp_frames = (
            FormantAmplitudeFrames.from_formant_harmonics_and_pitch_frames(
                formant_frames, pitch_harmonics, pitch_frames, lower=self.config.formants_amp_lower, upper=self.config.formants_amp_upper, rel_f0=self.config.formants_amp_rel_f0
            )
        )
        alpha_ratio_frames = AlphaRatioFrames.from_spec_frames(spec_frames, lower_band=self.config.alpha_ratio_lower_band, upper_band=self.config.alpha_ratio_upper_band)
        hammar_index_frames = HammarIndexFrames.from_spec_frames(spec_frames, pivot_point=self.config.hammar_index_pivot_point_freq, upper=self.config.hammar_index_upper_freq)
        spectral_slope_frames = SpectralSlopeFrames.from_spec_frames(spec_frames, bands=self.config.spectral_slopes_bands)
        mel_spec_frames = MelSpecFrames.from_spec_frames(spec_frames, n_mels=self.config.mel_spec_n_mels, lower=self.config.mel_spec_lower_freq, upper=self.config.mel_spec_upper_freq)
        mfcc_frames = MfccFrames.from_mel_spec_frames(mel_spec_frames, n_mfcc=self.config.mfcc_n, lifter=self.config.mfcc_lifter)
        spec_flux_frames = SpectralFluxFrames.from_spec_frames(spec_frames)
        rms_frames = RmsEnergyFrames.from_spec_frames(spec_frames)

        requirements = [
            audio_signal,
            pitch_frames,
            pulses_frames,
            jitter_frames,
            shimmer_frames,
            hnr_frames,
            formant_frames,
            pitch_harmonics,
            formant_amp_frames,
            alpha_ratio_frames,
            hammar_index_frames,
            spectral_slope_frames,
            mfcc_frames,
            spec_flux_frames,
            rms_frames,
        ]
        requirements_types = [type(r) for r in requirements]

        extracted_features = VoiceFeatures(frame=frame.tolist(), time=time.tolist())
        extracted_features.add_attributes(self.features.keys())

        for key, feat in self.features.items():
            for attr, req in feat.requires().items():
                idx = requirements_types.index(req)
                setattr(feat, attr, requirements[idx])

            self.logger.debug("Extracting feature %s", key)
            extracted_features.add_feature(key, feat.apply(time))

        return extracted_features


def cli():
    """Command line interface for extracting voice features.
    See `extract-voice -h` for details.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-f", "--filepath", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-t", "--time-step", type=float, dest="time_step")
    parser.add_argument("--skip-frames", type=int, default=1, dest="skip_frames")
    parser.add_argument("--config-filepath", type=optional_str, default=None, dest="config")

    args = parser.parse_args().__dict__

    if "config" in args:
        config = VoiceFeaturesConfig.from_yaml(args["config"])
    else:
        config = VoiceFeaturesConfig()

    extractor = VoiceExtractor(config=config)

    output = extractor.apply(
        args["filepath"], time_step=args["time_step"], skip_frames=args["skip_frames"]
    )

    output.write_json(
        os.path.join(
            args["outdir"],
            os.path.splitext(os.path.basename(args["filepath"]))[0]
            + "_voice_features.json",
        )
    )


if __name__ == "__main__":
    cli()

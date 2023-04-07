"""Extract voice features from an audio file.
"""

import argparse
import logging
import os
from typing import Dict, Optional, Union
import numpy as np
from scipy.interpolate import interp1d
from mexca.audio.features import (AlphaRatioFrames, AudioSignal, BaseFrames, FormantAmplitudeFrames, FormantFrames,
                                  HammarIndexFrames, HnrFrames, JitterFrames, MelSpecFrames, MfccFrames, PitchFrames,
                                  PitchHarmonicsFrames, PitchPulseFrames, RmsEnergyFrames, ShimmerFrames, SpecFrames,
                                  SpectralFluxFrames, SpectralSlopeFrames)
from mexca.data import VoiceFeatures
from mexca.utils import ClassInitMessage


class BaseFeature:
    """Base class for features.

    Can be used to create custom voice feature extraction classes.
    """

    def requires(self) -> Optional[Dict[str, type]]:
        """Specify objects required for feature extraction.

        This method can be overwritten to return a dictionary with keys as the names of objects
        required for computing features and values the types of these objects. The `VoiceExtractor`
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
    alpha_ratio_frames: Optional[AlphaRatioFrames] = None

    def requires(self) -> Optional[Dict[str, AlphaRatioFrames]]:
        return {"alpha_ratio_frames": AlphaRatioFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.alpha_ratio_frames.ts, self.alpha_ratio_frames.frames
        )(time)


class FeatureHammarIndex(BaseFeature):
    hammar_index_frames: Optional[HammarIndexFrames] = None

    def requires(self) -> Optional[Dict[str, HammarIndexFrames]]:
        return {"hammar_index_frames": HammarIndexFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.hammar_index_frames.ts, self.hammar_index_frames.frames
        )(time)


class FeatureSpectralSlope(BaseFeature):
    spectral_slope_frames: Optional[SpectralSlopeFrames] = None

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def requires(self) -> Optional[Dict[str, type]]:
        return {"spectral_slope_frames": SpectralSlopeFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        slope_idx = self.spectral_slope_frames.bands.index((self.lower, self.upper))
        # slope_idx = np.where(np.all(np.array(self.spectral_slope_frames.bands) == np.array([self.lower, self.upper])))
        return self._get_interp_fun(
            self.spectral_slope_frames.ts,
            self.spectral_slope_frames.frames[:, slope_idx],
        )(time)


class FeatureHarmonicDifference(BaseFeature):
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
    mfcc_frames: Optional[MfccFrames] = None

    def __init__(self, n_mfcc: int = 0) -> None:
        self.n_mfcc = n_mfcc

    def requires(self) -> Optional[Dict[str, MfccFrames]]:
        return {"mfcc_frames": MfccFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.mfcc_frames.ts, self.mfcc_frames.frames[:, self.n_mfcc]
        )(time)


class FeatureSpectralFlux(BaseFeature):
    spec_flux_frames: Optional[SpectralFluxFrames] = None

    def requires(self) -> Optional[Dict[str, SpectralFluxFrames]]:
        return {"spec_flux_frames": SpectralFluxFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(
            self.spec_flux_frames.ts, self.spec_flux_frames.frames
        )(time)


class FeatureRmsEnergy(BaseFeature):
    rms_frames: Optional[RmsEnergyFrames] = None

    def requires(self) -> Optional[Dict[str, type]]:
        return {"rms_frames": RmsEneryFrames}

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

    """

    def __init__(self, features: Optional[Dict[str, BaseFeature]] = None):
        self.logger = logging.getLogger("mexca.audio.extraction.VoiceExtractor")

        if features is None:
            features = self._set_default_features()

        self.features = features

        self.logger.debug(ClassInitMessage())

    @staticmethod
    def _set_default_features():
        return {
            "pitch_f0_hz": FeaturePitchF0(),
            "jitter_local_rel_f0": FeatureJitter(),
            "shimmer_local_rel_f0": FeatureShimmer(),
            "hnr_db": FeatureHnr(),
            "f1_freq_hz": FeatureFormantFreq(n_formant=0),
            "f1_bandwidth_hz": FeatureFormantBandwidth(n_formant=0),
            "f1_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=0),
            "f2_freq_hz": FeatureFormantFreq(n_formant=0),
            "f2_bandwidth_hz": FeatureFormantBandwidth(n_formant=1),
            "f2_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=1),
            "f3_freq_hz": FeatureFormantFreq(n_formant=1),
            "f3_bandwidth_hz": FeatureFormantBandwidth(n_formant=1),
            "f3_amplitude_rel_f0": FeatureFormantAmplitude(n_formant=1),
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
            audio_signal, frame_len=1024, hop_len=1024 // 4
        )

        spec_frames = SpecFrames.from_signal(
            audio_signal, frame_len=1024, hop_len=1024 // 4
        )
        pitch_frames = PitchFrames.from_signal(
            audio_signal, frame_len=1024, hop_len=1024 // 4
        )
        pulses_frames = PitchPulseFrames.from_signal_and_pitch_frames(
            audio_signal, pitch_frames
        )
        jitter_frames = JitterFrames.from_pitch_pulse_frames(pulses_frames)
        shimmer_frames = ShimmerFrames.from_pitch_pulse_frames(pulses_frames)
        hnr_frames = HnrFrames.from_frames(sig_frames)
        # TODO: Fix preemphasis for entire signal, not per frame
        formant_frames = FormantFrames.from_frames(sig_frames, preemphasis_from=None)
        pitch_harmonics = PitchHarmonicsFrames.from_spec_and_pitch_frames(
            spec_frames, pitch_frames, n_harmonics=100
        )
        formant_amp_frames = (
            FormantAmplitudeFrames.from_formant_harmonics_and_pitch_frames(
                formant_frames, pitch_harmonics, pitch_frames
            )
        )
        alpha_ratio_frames = AlphaRatioFrames.from_spec_frames(spec_frames)
        hammar_index_frames = HammarIndexFrames.from_spec_frames(spec_frames)
        spectral_slope_frames = SpectralSlopeFrames.from_spec_frames(spec_frames)
        mel_spec_frames = MelSpecFrames.from_spec_frames(spec_frames)
        mfcc_frames = MfccFrames.from_mel_spec_frames(mel_spec_frames)
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

    args = parser.parse_args().__dict__

    extractor = VoiceExtractor()

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

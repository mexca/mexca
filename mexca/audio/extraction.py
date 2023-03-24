"""Extract voice features from an audio file.
"""

import argparse
import logging
import os
from typing import Dict, Optional
import numpy as np
from scipy.interpolate import interp1d
from mexca.audio.features import (AudioSignal, BaseFrames, FormantAmplitudeFrames, FormantFrames, JitterFrames, HnrFrames,
                                  PitchFrames, PitchHarmonicsFrames, PitchPulseFrames, ShimmerFrames, SpecFrames)
from mexca.data import VoiceFeatures
from mexca.utils import ClassInitMessage


class BaseFeature:
    def requires(self) -> Optional[Dict[str, type]]:
        return None

    def _get_interp_fun(self, ts: np.ndarray, feature: np.ndarray) -> np.ndarray:
        return interp1d(ts, feature, kind="linear", bounds_error=False)

    def apply(self, time: np.ndarray) -> np.ndarray:
        return time


class FeaturePitchF0(BaseFeature):
    pitch_frames: PitchFrames = None

    def requires(self) -> Optional[Dict[str, type]]:
        return {"pitch_frames": PitchFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.pitch_frames.ts, self.pitch_frames.frames)(
            time
        )


class FeatureJitter(BaseFeature):
    jitter_frames: Optional[JitterFrames] = None

    def requires(self) -> Optional[Dict[str, type]]:
        return {"jitter_frames": JitterFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.jitter_frames.ts, self.jitter_frames.frames)(
            time
        )


class FeatureShimmer(BaseFeature):
    shimmer_frames: Optional[ShimmerFrames] = None

    def requires(self) -> Optional[Dict[str, type]]:
        return {"shimmer_frames": ShimmerFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.shimmer_frames.ts, self.shimmer_frames.frames)(
            time
        )


class FeatureHnr(BaseFeature):
    hnr_frames: Optional[HnrFrames] = None

    def requires(self) -> Optional[Dict[str, type]]:
        return {'hnr_frames': HnrFrames}

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.hnr_frames.ts, self.hnr_frames.frames)(time)


class FeatureFormantFreq(BaseFeature):
    formants: FormantFrames = None

    def __init__(self, n_formant: int):
        self.n_formant = n_formant

    def requires(self) -> Optional[Dict[str, type]]:
        return {"formants": FormantFrames}

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_freqs = self.formants.select_formant_attr(self.n_formant, 0)
        return self._get_interp_fun(self.formants.ts, formants_freqs)(time)


class FeatureFormantBandwidth(FeatureFormantFreq):
    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_bws = self.formants.select_formant_attr(self.n_formant, 1)
        return self._get_interp_fun(self.formants.ts, formants_bws)(time)


class FeatureFormantAmplitude(BaseFeature):
    formant_amp_frames: Optional[FormantAmplitudeFrames] = None

    def __init__(self, n_formant: int):
        self.n_formant = n_formant

    def requires(self) -> Optional[Dict[str, type]]:
        return {"formant_amp_frames": FormantAmplitudeFrames}

    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_amps = self.formant_amp_frames.frames
        return self._get_interp_fun(self.formant_amp_frames.ts, formants_amps[:, self.n_formant])(time)


class VoiceExtractor:
    """Extract voice features from an audio file.

    Currently, only the voice pitch as the fundamental frequency F0 can be extracted.
    The F0 is calculated using an autocorrelation function with a lower boundary of 75 Hz and an
    upper boudnary of 600 Hz. See the praat
    `manual <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch___.html>`_ for details.

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
            "pitch_f0": FeaturePitchF0(),
            "jitter_rel": FeatureJitter(),
            "shimmer_rel": FeatureShimmer(),
            "f1_freq": FeatureFormantFreq(n_formant=0),
            "f1_bandwidth": FeatureFormantBandwidth(n_formant=0),
            "f1_amplitude": FeatureFormantAmplitude(n_formant=0),
            "hnr": FeatureHnr()
        }

    def apply(
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
        formant_frames = FormantFrames.from_frames(sig_frames)
        pitch_harmonics = PitchHarmonicsFrames.from_spec_and_pitch_frames(
            spec_frames, pitch_frames, n_harmonics=100
        )
        formant_amp_frames = (
            FormantAmplitudeFrames.from_formant_harmonics_and_pitch_frames(
                formant_frames, pitch_harmonics, pitch_frames
            )
        )

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
        ]
        requirements_types = [type(r) for r in requirements]

        extracted_features = VoiceFeatures(frame=frame.tolist(), time=time.tolist())
        extracted_features.add_attributes(self.features.keys())

        for key, feat in self.features.items():
            for attr, req in feat.requires().items():
                idx = requirements_types.index(req)
                feat.__setattr__(attr, requirements[idx])

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

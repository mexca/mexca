"""Extract voice features from an audio file.
"""

import argparse
import logging
import os
from copy import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from mexca.audio.features import (AudioSignal, BaseFrames, FormantFrames, PitchFrames, PitchHarmonicsFrames,
                                  PitchPulseFrames, SpecFrames)
from mexca.data import VoiceFeatures
from mexca.utils import ClassInitMessage


class BaseFeature:
    def requires(self) -> Optional[Dict[str, type]]:
        return None


    def _get_interp_fun(self, ts: np.ndarray, feature: np.ndarray) -> np.ndarray:
        return interp1d(ts, feature, kind='linear', bounds_error=False)


    def apply(self, time: np.ndarray) -> np.ndarray:
        return time


class FeaturePitchF0(BaseFeature):
    pitch_frames: PitchFrames = None


    def requires(self) -> Optional[Dict[str, type]]:
        return {'pitch_frames': PitchFrames}


    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.pitch_frames.ts, self.pitch_frames.frames)(time)


class BasePitchPulsesFeature(BaseFeature):
    pitch_pulses: PitchPulseFrames = None


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
        return {'pitch_pulses': PitchPulseFrames}
    

    def _calc_period_length(self, pulses_idx: int) -> Tuple[List, np.ndarray]:
        # Calc period length as first order diff of pulse ts
        periods = np.diff(np.array([puls[0] for puls in self.pitch_pulses.frames[pulses_idx]]))

        # Filter out too short and long periods
        mask = np.logical_and(periods > self.lower, periods < self.upper)

        # Split periods according to mask and remove masked periods
        periods = np.array_split(periods[mask], np.where(~mask)[0])

        return periods, mask
    

    def _get_amplitude(self, pulses_idx: int) -> Tuple[List, List]:
        # Get amplitudes
        amps = np.array([puls[2] for puls in self.pitch_pulses.frames[pulses_idx]])[1:] # Skip first amplitude to align with periods

        # Calc period length and get mask for filtering amplitudes
        periods, mask = self._calc_period_length(pulses_idx)

        # Split periods according to mask and remove masked periods
        amps = np.array_split(amps[mask], np.where(~mask)[0])

        return amps, periods


    def _calc_feature(self):
        self._feature = np.array([self._calc_feature_frame(i) for i in range(len(self.pitch_pulses.frames))])


    def _calc_feature_frame(self, pulses_idx: int) -> float:
        return pulses_idx
    

    def apply(self, time: np.ndarray) -> np.ndarray:
        return self._get_interp_fun(self.pitch_pulses.ts, self.feature)(time)


class FeatureJitter(BasePitchPulsesFeature):
    def _calc_feature_frame(self, pulses_idx: int) -> float:
        if len(self.pitch_pulses.frames[pulses_idx]) > 0:
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
        if len(self.pitch_pulses.frames[pulses_idx]) > 0:
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
        

    def _select_formant_attr(self, n_attr: int = 0) -> np.ndarray:
        return np.array([f[self.n_formant][n_attr] if len(f) > self.n_formant else np.nan for f in self.formants.frames])


    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_freqs = self._select_formant_attr()
        return self._get_interp_fun(self.formants.ts, formants_freqs)(time)


class FeatureFormantBandwidth(FeatureFormantFreq):
    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_bws = self._select_formant_attr(1)
        return self._get_interp_fun(self.formants.ts, formants_bws)(time)


class FeatureFormantAmplitude(FeatureFormantFreq):
    harmonics: PitchHarmonicsFrames = None
    pitch_frames: PitchFrames = None


    def __init__(self, n_formant: int, lower: float = 0.8, upper: float = 1.2, rel_f0: bool = True):
        self.lower = lower
        self.upper = upper
        self.rel_f0 = rel_f0
        super().__init__(n_formant)


    def requires(self) -> Optional[Dict[str, type]]:
        return {'formants': FormantFrames, 'harmonics': PitchHarmonicsFrames, 'pitch_frames': PitchFrames}
    

    def _get_formant_amplitude(self, freqs: np.ndarray):
        f0 = self.pitch_frames.frames
        harmonic_freqs = f0[:, None] * (np.arange(self.harmonics.n_harmonics) + 1)[None, :]
        f0_amp = self.harmonics.frames[:, 0]
        freqs_lower = self.lower * freqs
        freqs_upper = self.upper * freqs
        freq_in_bounds = np.logical_and(harmonic_freqs > freqs_lower[:, None], harmonic_freqs < freqs_upper[:, None])
        harmonics_amp = copy(self.harmonics.frames)
        harmonics_amp[~freq_in_bounds] = np.nan
        harmonic_peaks = np.nanmax(harmonics_amp, axis=1)
        harmonic_peaks_db = 20 * np.log10(harmonic_peaks)

        if self.rel_f0:
            harmonic_peaks_db = harmonic_peaks_db - 20 * np.log10(f0_amp)

        return harmonic_peaks_db


    def apply(self, time: np.ndarray) -> Optional[np.ndarray]:
        formants_freqs = self._select_formant_attr()
        formants_amps = self._get_formant_amplitude(formants_freqs)
        return self._get_interp_fun(self.formants.ts, formants_amps)(time)


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
            'f1_freq': FeatureFormantFreq(n_formant=0),
            'f1_bandwidth': FeatureFormantBandwidth(n_formant=0),
            'f1_amplitude': FeatureFormantAmplitude(n_formant=0),
            # 'loudness': FeatureLoudness()
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
        
        sig_frames = BaseFrames.from_signal(audio_signal, frame_len=1024, hop_len=1024 // 4)

        spec_frames = SpecFrames.from_signal(audio_signal, frame_len=1024, hop_len=1024 // 4)
        pitch_frames = PitchFrames.from_signal(audio_signal, frame_len=1024, hop_len=1024 // 4)
        pitch_pulses = PitchPulseFrames.from_signal_and_pitch(audio_signal, pitch_frames)
        formant_frames = FormantFrames.from_frames(sig_frames)
        pitch_harmonics = PitchHarmonicsFrames.from_spec_and_pitch(spec_frames, pitch_frames, n_harmonics=100)

        requirements = [audio_signal, pitch_frames, pitch_pulses, formant_frames, pitch_harmonics]
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
    
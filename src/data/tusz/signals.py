"""Module for loading data from edf files"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyedflib
from scipy.signal import resample

################################################################################
# DATA LOADING


def get_sampled_signals_and_names(edf_path: Path, sampling_rate: int) -> Tuple[np.ndarray, List[str]]:
    """Read ``.edf`` file and retrieve EEG scans and relative information.

    Args:
        edf_path (Path): Path to ``.edf`` file.

    Raises:
        AssetionError: on unexpected input

    Returns:
        Tuple[np.ndarray, List[str]]: Return three terms:
            - Array of EEG scans of shape ``(nb_channels, nb_samples)``
            - List of channels names
    """
    edf_reader = pyedflib.EdfReader(str(edf_path))

    signals, signal_channels, input_sampling_rate = read_eeg_signals(edf_reader)

    # Resample to target rate in Hz = samples/sec. Do nothing if already at required freq
    if sampling_rate < input_sampling_rate:
        out_num = int(signals.shape[1] / input_sampling_rate * sampling_rate)
        signals = resample(signals, num=out_num, axis=1)
    elif sampling_rate > input_sampling_rate:
        raise ValueError(f"Required sampling rate {sampling_rate} higher than file rate {input_sampling_rate}")

    return signals, signal_channels


def read_eeg_signals(edf_reader: pyedflib.EdfReader) -> Tuple[np.ndarray, List[str], int]:
    """Get EEG signals and names from  edf file

    Args:
        edf_reader (pyedflib.EdfReader): EDF reader

    Raises:
        AssertionError: On invalid data, see messages

    Returns:
        Tuple[np.ndarray, List[str], int]: signals, channel_names, sampling_rate
    """

    signal_channels = edf_reader.getSignalLabels()
    n_channels = edf_reader.signals_in_file

    if n_channels != len(signal_channels):
        raise AssertionError(f"Number of channels different from names: {n_channels} != {len(signal_channels)}")

    # nb_samples is an array of nb_samples per channel
    nb_samples = edf_reader.getNSamples()

    sampling_rates = edf_reader.getSampleFrequencies()

    signals = []
    signal_chnls_f = []
    ref_rate = None

    for i, (ch_name, ch_samples, ch_rate) in enumerate(zip(signal_channels, nb_samples, sampling_rates)):
        if ch_name.startswith("EEG"):
            if not signals:
                ref_samples = ch_samples
                ref_rate = int(ch_rate)

            assert ch_samples == ref_samples, f"Channel '{ch_name}' has lenght {ch_samples}, expecting {ref_samples}"

            assert np.allclose(np.modf(ch_rate)[0], 0), f"Non-integer sampling rate in {ch_name}: {ch_rate}"
            assert ch_rate == ref_rate, f"Channel '{ch_name}' has sampling rate {ch_rate}, expecting {ref_rate}"

            signal_chnls_f.append(ch_name)
            signals.append(edf_reader.readSignal(i))

    return np.array(signals), signal_chnls_f, ref_rate


def extract_segment(signal: np.ndarray, start_time: float, end_time: float, sampling_rate: float) -> np.ndarray:
    """Split time-array using time stamps, given sampling rate."""
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    return signal[:, start_idx:end_idx]

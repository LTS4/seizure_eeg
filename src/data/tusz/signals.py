"""Module for loading data from edf files"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyedflib
from scipy.signal import resample

################################################################################
# DATA LOADING


def get_signals_and_info(edf_path: Path, sampling_rate: int) -> Tuple[np.ndarray, List[str]]:
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

    signal_channels = edf_reader.getSignalLabels()
    signals = read_eeg_signals(edf_reader)

    input_sampling_rate = get_sampling_rate(edf_reader)

    # Resample to target rate in Hz = samples/sec. Do nothing if already at required freq
    if sampling_rate < input_sampling_rate:
        out_num = int(signals.shape[1] / input_sampling_rate * sampling_rate)
        signals = resample(signals, num=out_num, axis=1)
    elif sampling_rate > input_sampling_rate:
        raise ValueError(f"Required sampling rate {sampling_rate} higher than file rate {input_sampling_rate}")

    return signals, signal_channels


def read_eeg_signals(edf_reader: pyedflib.EdfReader) -> np.ndarray:
    """
    Get EEG signals in edf file

    Args:
        edf: edf object

    Raises:
        AssetionError: on unexpected input

    Returns:
        signals: shape (num_channels, num_data_points)

    (c) 2021 Siyi Tang
    """
    n_channels = edf_reader.signals_in_file

    # samples is an array of nb_samples per channel
    nb_samples = edf_reader.getNSamples()
    assert np.all(nb_samples == nb_samples[0]), f"Found samples of different lenghts: {nb_samples}"

    signals = np.zeros((n_channels, nb_samples[0]))

    for i in range(n_channels):
        # TODO: this could raise
        signals[i, :] = edf_reader.readSignal(i)
    return signals


################################################################################
# SAMPLING RATE


def get_sampling_rate(edf_reader: pyedflib.EdfReader) -> float:
    """Get the unique sampling rate of the signals.

    Args:
        edf_reader (pyedflib.EdfReader): EDF reader

    Raises:
        AssetionError: on unexpected input

    Returns:
        float: sampling rate
    """
    sampling_rates = edf_reader.getSampleFrequencies()
    sampling_rate0 = sampling_rates[0]

    assert np.all(sampling_rates == sampling_rate0), "Found different sampling rates in same file"
    assert np.allclose(np.modf(sampling_rate0)[0], 0), "Fount noninteger sampling rate"

    return int(sampling_rate0)


def extract_segment(signal: np.ndarray, start_time: float, end_time: float, sampling_rate: float) -> np.ndarray:
    """Split time-array using time stamps, given sampling rate."""
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    return signal[:, start_idx:end_idx]

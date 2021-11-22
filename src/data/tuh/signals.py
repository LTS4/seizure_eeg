"""Module for loading data from edf files"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyedflib


def get_signals_and_info(edf_path: Path) -> Tuple[np.ndarray, List[str], float]:
    """Read ``.edf`` file and retrieve EEG scans and relative information.

    Args:
        edf_path (Path): Path to ``.edf`` file.

    Returns:
        Tuple[np.ndarray, List[str], float]: Return three terms:
            - Array of EEG scans of shape ``(nb_channels, nb_samples)``
            - List of channels names
            - Sampling rate
    """
    edf_reader = pyedflib.EdfReader(str(edf_path))

    signal_channels = edf_reader.getSignalLabels()
    signals = read_eeg_signals(edf_reader)
    sampling_rate = get_sampling_rate(edf_reader)

    return signals, signal_channels, sampling_rate


def get_sampling_rate(edf_reader: pyedflib.EdfReader) -> float:
    """Get the unique sampling rate of the signals.

    Args:
        edf_reader (pyedflib.EdfReader): EDF reader

    Returns:
        float: sampling rate
    """
    sampling_rates = edf_reader.getSampleFrequencies()
    assert np.all(sampling_rates == sampling_rates[0]), "Different sampling rates in same file"
    return sampling_rates[0]


def read_eeg_signals(edf_reader: pyedflib.EdfReader) -> np.ndarray:
    """
    Get EEG signals in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)

    (c) 2021 Siyi Tang
    """
    n_channels = edf_reader.signals_in_file

    # samples is an array of nb_samples per channel
    samples = edf_reader.getNSamples()
    assert np.all(samples == samples[0])

    signals = np.zeros((n_channels, samples[0]))

    for i in range(n_channels):
        # TODO: this could raise
        signals[i, :] = edf_reader.readSignal(i)
    return signals

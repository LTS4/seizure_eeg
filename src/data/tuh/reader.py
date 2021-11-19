import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import resample

from config import PATH_TUH_RAW


SUFFIXES = [".edf", ".tse", ".lbl"]
SUFFIXES_BINARY = [".edf", ".tse_bi", ".lbl_bi"]


def list_all_edf_files(root_path: Path) -> List[Path]:
    filelist = [
        Path(dirpath) / file
        for dirpath, _, filenames in os.walk(root_path)
        for file in filenames
        if file.endswith("edf")
    ]

    return filelist


def read_edf_and_labels(edf_path: Path, binary: bool):
    edf_reader = pyedflib.EdfReader(str(edf_path))

    signal_labels = edf_reader.getSignalLabels()
    signals = get_edf_signals(edf_reader)


def get_edf_annotations(edf_path: Path, binary: bool) -> :
    raise NotImplementedError


def get_edf_signals(edf_reader: pyedflib.EdfReader) -> np.ndarray:
    """
    Get EEG signal in edf file
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
        signals[i,:] = edf_reader.readSignal(i)
    return signals


if __name__ == "__main__":
    file = list_all_edf_files(PATH_TUH_RAW)[0]
    read_edf_and_labels(file, False)

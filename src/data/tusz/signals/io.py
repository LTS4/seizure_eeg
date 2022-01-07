"""I/O functions for EDF signals"""
import re
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyedflib
from pandera import check_types
from pandera.typing import DataFrame

from src.data.schemas import SignalsDF
from src.data.tusz.constants import CHANNELS, REGEX_SIGNAL_CHANNELS


def fix_channel_name(name: str) -> str:
    """Fix channel name if it has a known format.

    Example:
        - change "EEG {ch}-LE" to "EEG {ch}-REF"
    """
    return name.replace("-LE", "-REF")


def format_channel_names(names: List[str]) -> List[str]:
    return [
        match.group("ch") if match else None
        for name in names
        for match in (re.match(REGEX_SIGNAL_CHANNELS, name),)
    ]


# @lru_cache
@check_types
def read_eeg_signals(edf_path: Path) -> Tuple[DataFrame[SignalsDF], int]:
    """Get EEG signals and names from  edf file

    Args:
        edf_reader (pyedflib.EdfReader): EDF reader

    Raises:
        AssertionError: On invalid data, see messages

    Returns:
        Tuple[DataFrame, int]: signals, sampling_rate
    """
    try:
        edf_reader = pyedflib.EdfReader(str(edf_path))
    except OSError as err:
        raise OSError(f"Error from file {edf_path}") from err

    signal_channels = format_channel_names(edf_reader.getSignalLabels())
    n_channels = edf_reader.signals_in_file

    if n_channels != len(signal_channels):
        raise AssertionError(
            f"Number of channels different from names: {n_channels} != {len(signal_channels)}"
        )

    # nb_samples is an array of nb_samples per channel
    nb_samples = edf_reader.getNSamples()

    sampling_rates = edf_reader.getSampleFrequencies()

    # Prepare list to be read and validate metadata
    try:
        to_read = [signal_channels.index(chl) for chl in CHANNELS]
    except ValueError as err:
        raise ValueError(f"File {edf_path} does not contain all needed channels") from err

    nb_samples = nb_samples[to_read]
    assert np.allclose(
        nb_samples, nb_samples[0]
    ), f"EEG channels with different lenght in {edf_path}"

    sampling_rates = sampling_rates[to_read]
    assert np.allclose(np.modf(sampling_rates)[0], 0), f"Non-integer sampling rate in {edf_path}"
    assert np.allclose(
        sampling_rates, sampling_rates[0]
    ), f"EEG channels with different sampling rates in {edf_path}"
    ref_rate = sampling_rates[0]

    # signals[ch_name] = edf_reader.readSignal(i)

    signals = pd.DataFrame(
        data=np.array([edf_reader.readSignal(i) for i in to_read]).T,
        columns=CHANNELS,
    )

    return signals, ref_rate


def read_signal(edf_reader: pyedflib.EdfReader, ch_idx: int):
    return edf_reader.readSignal(ch_idx)

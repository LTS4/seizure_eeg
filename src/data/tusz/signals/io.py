"""I/O functions for EDF signals"""
import re
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

    signals = pd.DataFrame()
    ref_rate = None

    for i, (ch_name, ch_samples, ch_rate) in enumerate(
        zip(signal_channels, nb_samples, sampling_rates)
    ):
        if ch_name in CHANNELS:
            if ref_rate is None:
                ref_samples = ch_samples
                ref_rate = int(ch_rate)

            assert (
                ch_samples == ref_samples
            ), f"Channel '{ch_name}' has length {ch_samples}, expecting {ref_samples}"

            assert np.allclose(
                np.modf(ch_rate)[0], 0
            ), f"Non-integer sampling rate in {ch_name}: {ch_rate}"
            assert (
                ch_rate == ref_rate
            ), f"Channel '{ch_name}' has sampling rate {ch_rate}, expecting {ref_rate}"

            signals[fix_channel_name(ch_name)] = edf_reader.readSignal(i)

    return signals, ref_rate

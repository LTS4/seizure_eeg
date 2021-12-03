"""Processing data from edf files"""

from typing import List

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Index, Series
from scipy.signal import resample

from src.data.schemas import SignalsDF
from src.data.tusz.constants import TEMPLATE_SIGNAL_CHANNELS

################################################################################
# DATA LOADING


def get_resampled_signals(
    signals: DataFrame[SignalsDF], input_sampling_rate: int, sampling_rate: int
) -> DataFrame[SignalsDF]:
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

    # Resample to target rate in Hz = samples/sec. Do nothing if already at required freq
    if sampling_rate < input_sampling_rate:
        out_num = int(len(signals) / input_sampling_rate * sampling_rate)
        signals = pd.DataFrame(resample(signals, num=out_num, axis=0), columns=signals.columns)
    elif sampling_rate > input_sampling_rate:
        raise ValueError(f"Required sampling rate {sampling_rate} higher than file rate {input_sampling_rate}")

    return signals


def extract_segment(signal: np.ndarray, start_time: float, end_time: float, sampling_rate: float) -> np.ndarray:
    """Split time-array using time stamps, given sampling rate."""
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    return signal[:, start_idx:end_idx]


def pairwise_diff(diff_label: str, signals: DataFrame[SignalsDF]) -> Series[float]:
    el1, el2 = diff_label.split("-")
    return signals[TEMPLATE_SIGNAL_CHANNELS.format(el1)] - signals[TEMPLATE_SIGNAL_CHANNELS.format(el2)]


def get_diff_signals_buggy(signals: DataFrame[SignalsDF], label_channels: Index[str]):
    """This should be a fater version of get_diff_signals"""
    lhs, rhs = [], []

    for diff_label in label_channels:
        el1, el2 = diff_label.split("-")
        lhs.append(TEMPLATE_SIGNAL_CHANNELS.format(el1))
        rhs.append(TEMPLATE_SIGNAL_CHANNELS.format(el2))

    return pd.DataFrame((signals[lhs] - signals[rhs]).values, columns=label_channels)


def get_diff_signals(signals: DataFrame[SignalsDF], label_channels: List[str]):
    """Take as input a signals dataframe and return the columm differences specified in *label_channels*"""
    loc_signals = pd.DataFrame(np.empty((len(signals), len(label_channels))), columns=label_channels)

    for diff_label in label_channels:
        el1, el2 = diff_label.split("-")
        loc_signals[diff_label] = (
            signals[TEMPLATE_SIGNAL_CHANNELS.format(el1)] - signals[TEMPLATE_SIGNAL_CHANNELS.format(el2)]
        )

    return loc_signals

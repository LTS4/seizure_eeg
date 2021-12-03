"""Processing data from edf files"""
from typing import List, Optional

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Index
from scipy.signal import resample

from src.data.schemas import SignalsDF
from src.data.tusz.constants import TEMPLATE_SIGNAL_CHANNELS

################################################################################
# RASAMPLING


def get_resampled_signals(
    signals: DataFrame, sampling_rate_in: int, sampling_rate_out: int
) -> DataFrame:
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
    if sampling_rate_out < sampling_rate_in:
        out_num = int(len(signals) / sampling_rate_in * sampling_rate_out)
        signals = pd.DataFrame(resample(signals, num=out_num, axis=0), columns=signals.columns)
    elif sampling_rate_out > sampling_rate_in:
        raise ValueError(
            f"Required sampling rate {sampling_rate_out} higher than file rate {sampling_rate_in}"
        )

    return signals


####################################################################################################
# PAIRWISE DIFFERENCES


def get_diff_signals_buggy(signals: DataFrame[SignalsDF], label_channels: Index[str]):
    """This should be a fater version of get_diff_signals"""
    lhs, rhs = [], []

    for diff_label in label_channels:
        el1, el2 = diff_label.split("-")
        lhs.append(TEMPLATE_SIGNAL_CHANNELS.format(el1))
        rhs.append(TEMPLATE_SIGNAL_CHANNELS.format(el2))

    return pd.DataFrame((signals[lhs] - signals[rhs]).values, columns=label_channels)


def get_diff_signals(signals: DataFrame[SignalsDF], label_channels: List[str]):
    """Take as input a signals dataframe and return the columm differences specified in
    *label_channels*"""
    loc_signals = pd.DataFrame(
        np.empty((len(signals), len(label_channels))), columns=label_channels
    )

    for diff_label in label_channels:
        el1, el2 = diff_label.split("-")
        loc_signals[diff_label] = (
            signals[TEMPLATE_SIGNAL_CHANNELS.format(el1)]
            - signals[TEMPLATE_SIGNAL_CHANNELS.format(el2)]
        )

    return loc_signals


####################################################################################################
# PIPELINE


def process_signals(
    signals: DataFrame,
    sampling_rate_in: int,
    sampling_rate_out: int,
    diff_labels: Optional[Index[str]] = None,
) -> DataFrame:
    """Process signals read from edf file

    Args:
        signals (DataFrame): Dataframe of signals of shape ``nb_samples x nb_channels``
        sampling_rate_in (int): Sampling rate as read from edf file
        sampling_rate_out (int): Desired sampling rate
        diff_labels (Optional[Index[str]], optional): Labels defining which columns shall be
            subtracted to generate final signals. Defaults to None, in which case the full EEG
            dadaframe is returned.

    Returns:
        DataFrame: Processed signals
    """

    if diff_labels is not None:
        signals = get_diff_signals(signals, diff_labels)

    signals = get_resampled_signals(signals, sampling_rate_in, sampling_rate_out)

    return signals

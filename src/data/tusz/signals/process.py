"""Processing data from edf files"""
from typing import List

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Index
from scipy.signal import resample

from src.data.schemas import SignalsDF, SignalsDiffDF
from src.data.tusz.constants import SIGNAL_CHANNELS_FMT

################################################################################
# RASAMPLING


def resample_signals(
    signals: np.ndarray, sampling_rate_in: int, sampling_rate_out: int
) -> np.ndarray:
    """Resample signals dataframe

    Args:
        signals (np.ndarray): array of time series, with time on axis 0
        sampling_rate_in (int): sampling rate of input data
        sampling_rate_out (int): desired sampling rate

    Raises:
        ValueError: If *sampling_rate_out* is greater than *sampling_rate_in*

    Returns:
        np.ndarray: Resampled array
    """

    # Resample to target rate in Hz = samples/sec. Do nothing if already at required freq
    if sampling_rate_out < sampling_rate_in:
        out_num = int(len(signals) / sampling_rate_in * sampling_rate_out)
        signals = resample(signals, num=out_num, axis=0)
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
        lhs.append(SIGNAL_CHANNELS_FMT.format(el1))
        rhs.append(SIGNAL_CHANNELS_FMT.format(el2))

    return pd.DataFrame((signals[lhs] - signals[rhs]).values, columns=label_channels)


def get_diff_signals(
    signals: DataFrame[SignalsDF], label_channels: List[str]
) -> DataFrame[SignalsDiffDF]:
    """Take as input a signals dataframe and return the columm differences specified in
    *label_channels*"""
    diff_signals = pd.DataFrame(
        np.empty((len(signals), len(label_channels))), columns=label_channels
    )

    for diff_label in label_channels:
        el1, el2 = diff_label.split("-")
        diff_signals.loc[:, diff_label] = (signals[el1] - signals[el2]).values

    return diff_signals


####################################################################################################
# PIPELINE


def preprocess_signals(
    signals: DataFrame[SignalsDF],
    sampling_rate_in: int,
    sampling_rate_out: int,
) -> DataFrame[SignalsDF]:
    """Pre-process signals read from edf file.

    Processing steps:
        1. Resample signals

    Args:
        signals (DataFrame): Dataframe of signals of shape ``nb_samples x nb_channels``
        sampling_rate_in (int): Sampling rate as read from edf file
        sampling_rate_out (int): Desired sampling rate

    Returns:
        DataFrame: Processed signals
    """

    # 1. Resample signals
    signals = pd.DataFrame(
        data=resample_signals(signals, sampling_rate_in, sampling_rate_out),
        columns=signals.columns,
    )

    return signals

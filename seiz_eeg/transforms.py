# pylint: disable=too-few-public-methods
"""Transformations functions and classes to use in pipelines"""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from seiz_eeg.schemas import SignalsDF, SignalsDiffDF


class Concatenate:
    """Apply list of functions one after the other"""

    def __init__(self, func_list: List[Callable[[Any], Any]]) -> None:
        self.func_list = func_list

    def __call__(self, *args: Any) -> Any:
        for func in self.func_list:
            args = func(*args)

        return args


class SplitWindows:
    """Split windows"""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def __call__(self, signals: np.ndarray) -> np.ndarray:
        """Split windows"""

        nb_win = signals.shape[0] // self.window_size

        return signals[: nb_win * self.window_size].reshape(
            nb_win,  # nb of windows
            self.window_size,  # nb of samples per window (time axis)
            signals.shape[1],  # nb of signals
        )


class ExtractFromAxis:
    """Extract slice of tensor from given axis"""

    def __init__(self, axis: int, extremes: Tuple[Optional[int], Optional[int]]) -> None:
        self.axis = axis
        self.extremes = extremes

    def __call__(self, X: np.ndarray) -> Any:
        # Define slices to extract
        extractor = len(X.shape) * [slice(None)]
        extractor[self.axis] = slice(*self.extremes)

        return X[extractor]


class OldTransform:
    """Sequence of transforms from previous code version"""

    def __init__(
        self,
        window_size: Optional[int] = None,
        fft_coeffs: Optional[Tuple[int, int]] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.window_size = window_size
        self.fft_coeffs = fft_coeffs

        self.mean = mean
        self.std = std

    def __call__(
        self,
        signals: np.ndarray,
    ) -> Any:
        # 3. Optional: Split windows
        if self.window_size:
            time_axis = 1

            signals = signals.reshape(
                signals.shape[0] // self.window_size,  # nb of windows
                self.window_size,  # nb of samples per window (time axis)
                signals.shape[1],  # nb of signals
            )
        else:
            time_axis = 0

        # 3. Optional: Compute fft
        if self.fft_coeffs:
            # Define slices to extract
            extractor = len(signals.shape) * [slice(None)]
            extractor[time_axis] = slice(*self.fft_coeffs)

            # Actual fft
            signals = np.abs(np.fft.rfft(signals, axis=time_axis))
            signals = signals[extractor]

        # Center data. This is always performed, except for `_compute_mean`
        if hasattr(self, "mean") and self.mean is not None:
            signals -= self.mean

            # Normalize data. This is always performed, except for `_compute_std`
            if hasattr(self, "std") and self.std is not None:
                signals /= self.std

        return signals


####################################################################################################
# PAIRWISE DIFFERENCES


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

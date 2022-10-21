"""EEG Data class with common data retrieval"""
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pandera import check_types
from pandera.typing import DataFrame
from tqdm import tqdm

from seiz_eeg.constants import EEG_CHANNELS, EEG_MONTAGES
from seiz_eeg.schemas import ClipsDF
from seiz_eeg.transforms import SplitWindows
from seiz_eeg.tusz.signals.io import read_parquet
from seiz_eeg.tusz.signals.process import get_diff_signals


def _identity(x):
    return x


class EEGDataset:
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        clips_df: DataFrame[ClipsDF],
        *,
        signal_transform: Optional[Callable[[NDArray[np.float_]], Union[NDArray, Any]]] = None,
        label_transform: Optional[Callable[[int], Any]] = None,
        diff_channels: bool = False,
    ) -> None:
        """Dataset of EEG clips with seizure labels

        Args:
            clips_df (DataFrame[ClipsDF]): Pandas dataframe of EEG clips annotations
            diff_channels (bool, optional): Whether to use channel differences
                or not. Defaults to False.
            seed (int, optional): Random seed. Defaults to None.
        """
        super().__init__()

        logging.debug("Creating clips from segments")
        self.clips_df = clips_df

        # We compute the lenght of each segment
        lenghts = np.unique(self.clips_df[ClipsDF.end_time] - self.clips_df[ClipsDF.start_time])
        # We check whether lenghts is unique, since floating error may occur
        self.clip_lenght = lenghts[0] if np.allclose(lenghts, lenghts[0]) else -1

        self.s_rate = clips_df[ClipsDF.sampling_rate].unique().item()
        self._clip_size = round(self.clip_lenght * self.s_rate)

        self.diff_channels = diff_channels

        self.signal_transform = signal_transform or _identity
        self.label_transform = label_transform or _identity

        self.output_shape = self._get_output_shape()

    def __getitem__(self, index: int) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
        label, start_time, end_time, _, s_rate, signals_path = self.clips_df.iloc[index]

        start_sample = int(start_time * s_rate)

        if self.clip_lenght > 0:
            assert np.allclose(end_time - start_time, self.clip_lenght)
            # We use clip_lenght instead of end_time to avoid floating point errors
            end_sample = start_sample + self._clip_size
        else:
            # In this case we return segments instead of clips, note they have different lengths
            end_sample = int(end_time * s_rate)

        signals = read_parquet(signals_path).iloc[start_sample:end_sample]
        assert 0 not in signals.shape

        # 1. (opt) Subtract pairwise columns
        if self.diff_channels:
            signals = get_diff_signals(signals, EEG_MONTAGES).values
        else:
            signals = signals.values

        return self.signal_transform(signals), self.label_transform(label)

    def get_label_array(self) -> np.ndarray:
        return self.clips_df[ClipsDF.label].map(self.label_transform).values

    def get_channels_names(self) -> List[str]:
        if self.diff_channels:
            return EEG_MONTAGES
        else:
            return EEG_CHANNELS

    def _get_output_shape(self) -> Tuple[tuple, tuple]:
        X0, y0 = self.__getitem__(0)
        return X0.shape, y0.shape

    def __len__(self) -> int:
        return len(self.clips_df)


class EEGFileDataset(EEGDataset):
    """Extension of :class:`EEGDataset` which returns a tensor of all clips from the same file."""

    def __init__(
        self,
        clips_df: DataFrame[ClipsDF],
        *,
        signal_transform: Optional[Callable[[NDArray[np.float_]], Union[NDArray, Any]]] = None,
        label_transform: Optional[Callable[[int], Any]] = None,
        diff_channels: bool = False,
    ) -> None:
        super().__init__(
            clips_df,
            signal_transform=signal_transform,
            label_transform=label_transform,
            diff_channels=diff_channels,
        )

        self.session_ids = self.clips_df.index.unique(level="session")

        self._split_clips = SplitWindows(self._clip_size)

    def __len__(self) -> int:
        return len(self.session_ids)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        session: DataFrame[ClipsDF] = self.clips_df.xs(
            self.session_ids[index], level=ClipsDF.session
        )

        clip_indices = session.index.get_level_values(ClipsDF.segment)
        nb_clips = clip_indices.max() + 1

        labels = session[ClipsDF.label].map(self.label_transform).values

        end_time, signals_path = session.iloc[-1][[ClipsDF.end_time, ClipsDF.signals_path]]

        end_sample = nb_clips * self._clip_size
        assert (
            -1 <= end_sample - end_time * self.s_rate <= 1
        ), f"Discrepancy in lenghts for session #{index}"

        signals = read_parquet(signals_path).iloc[:end_sample]

        # 1. (opt) Subtract pairwise columns
        if self.diff_channels:
            signals = get_diff_signals(signals, EEG_MONTAGES).values
        else:
            signals = signals.values

        # 3. Clip and keep only labelled ones
        signals = self._split_clips(signals)[clip_indices]

        return self.signal_transform(signals), labels

    def _getclip(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return super().__getitem__(index)

    def _get_output_shape(self) -> Tuple[tuple, tuple]:
        X0, y0 = self._getclip(0)
        X0.unsqueeze_(dim=0)
        y0.unsqueeze_(dim=0)

        return X0.shape, y0.shape


def to_arrays(data: EEGDataset, pbar=False) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """Load all signals from :arg:`data` into a tensor.

    Args:
        data (EEGDataset): Clips dataset.
            All clips must have the same lenght to be stacked.
        pbar (bool, optional): Wether toshow a progress bar while loading. Defaults to False.

    Returns:
        Tuple[NDArray[np.float_], NDArray[np.int_]]: Signals and labels
    """
    # return next(iter(DataLoader(data, batch_size=len(data))))
    if pbar:
        data = tqdm(data)
    x, y = zip(*([x, y] for x, y in data))
    try:
        x = np.stack(x)
    except ValueError:
        pass

    return x, np.stack(y)

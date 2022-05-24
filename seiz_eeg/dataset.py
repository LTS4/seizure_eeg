"""EEG Data class with common data retrieval"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pandas import IndexSlice as idx
from pandera import check_types
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from seiz_eeg.clips import make_clips
from seiz_eeg.constants import EEG_CHANNELS, EEG_MONTAGES, GLOBAL_CHANNEL
from seiz_eeg.schemas import ClipsDF
from seiz_eeg.transforms import SplitWindows
from seiz_eeg.tusz.signals.io import read_parquet
from seiz_eeg.tusz.signals.process import get_diff_signals


class EEGDataset(Dataset):
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        segments_df: DataFrame[ClipsDF],
        *,
        clip_length: float,
        clip_stride: Union[float, str],
        overlap_action: str = "ignore",
        diff_channels: bool = False,
        node_level: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """Dataset of EEG clips with seizure labels

        Args:
            segments_df (DataFrame[ClipsDF]): Pandas dataframe of EEG semgnets annotations
            clip_length (float): Clip lenght for :func:`make_clips`
            clip_stride (Union[float, str]): Clip stride for :func:`make_clips`
            overlap_action (str, optional): Overlap action for
                :func:`make_clips`. Defaults to 'ignore'.
            diff_channels (bool, optional): Whether to use channel differences
                or not. Defaults to False.
            node_level (bool, optional): Wheter to get node-level or global
                labels (only latter is currently supported). Defaults to False.
            device (Optional[str], optional): Torch device. Defaults to None.
        """
        super().__init__()

        logging.debug("Creating clips from segments")
        self.clips_df = make_clips(
            segments_df,
            clip_length=clip_length,
            clip_stride=clip_stride,
            overlap_action=overlap_action,
        )
        self._clip_lenght = clip_length

        self.s_rate = segments_df[ClipsDF.sampling_rate].unique().item()
        self._clip_size = int(self._clip_lenght * self.s_rate)

        self.diff_channels = diff_channels
        self.node_level(node_level)
        self.device = device

        self.output_shape = self._get_output_shape()

    def node_level(self, node_level: bool):
        """Setter for the node-level labels retrieval"""
        self._node_level = node_level

        if node_level:
            raise NotImplementedError
            # self._clips_df = self.clips_df.drop(GLOBAL_CHANNEL).groupby(AnnotationDF.channel)
        else:
            self._clips_df: DataFrame[ClipsDF] = self.clips_df.loc[idx[:, :, :, GLOBAL_CHANNEL]]

    def _get_from_df(
        self, index: int
    ) -> Tuple[Union[int, List[int]], float, float, np.datetime64, int, str]:
        if self._node_level:
            raise NotImplementedError

        return self._clips_df.iloc[index]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, start_time, end_time, _, s_rate, signals_path = self._get_from_df(index)

        start_sample = int(start_time * s_rate)

        if self._clip_lenght > 0:
            assert np.allclose(end_time - start_time, self._clip_lenght)
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

        # 2. Convert to torch
        signals = torch.tensor(signals, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)

        return signals, label

    def get_label_array(self) -> np.ndarray:
        return self._clips_df[ClipsDF.label].values

    def get_channels_names(self) -> List[str]:
        if self.diff_channels:
            return EEG_MONTAGES
        else:
            return EEG_CHANNELS

    def _get_output_shape(self) -> Tuple[torch.Size, torch.Size]:
        X0, y0 = self.__getitem__(0)
        return X0.shape, y0.shape

    def __len__(self) -> int:
        return len(self._clips_df)


class EEGFileDataset(EEGDataset):
    """Extension of :class:`EEGDataset` which returns a tensor of all clips from the same file."""

    def __init__(
        self,
        segments_df: DataFrame[ClipsDF],
        *,
        clip_length: float,
        overlap_action: str = "ignore",
        diff_channels: bool = False,
        node_level: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            segments_df,
            clip_length=clip_length,
            clip_stride=clip_length,
            overlap_action=overlap_action,
            diff_channels=diff_channels,
            node_level=node_level,
            device=device,
        )

        self.session_ids = self._clips_df.index.unique(level="session")

        self._split_clips = SplitWindows(self._clip_size)

    def _getclip(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.session_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session: DataFrame[ClipsDF] = self._clips_df.loc[
            (slice(None), self.session_ids[index]),
        ]
        clip_indices = session.index.get_level_values(ClipsDF.segment)
        nb_clips = clip_indices.max() + 1

        labels = torch.as_tensor(session[ClipsDF.label], dtype=torch.long, device=self.device)

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

        # 2. Convert to torch
        signals = torch.tensor(signals, dtype=torch.float32, device=self.device)

        # 3. Clip and keep only labelled ones
        signals = self._split_clips(signals)[clip_indices]

        return signals, labels

    def _get_output_shape(self) -> Tuple[torch.Size, torch.Size]:
        X0, y0 = self._getclip(0)
        X0.unsqueeze_(dim=0)
        y0.unsqueeze_(dim=0)

        return X0.shape, y0.shape

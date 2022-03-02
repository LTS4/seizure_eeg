"""EEG Data class with common data retrieval"""
import logging
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from pandas import IndexSlice as idx
from pandera import check_types
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from seiz_eeg.constants import EEG_CHANNELS, EEG_MONTAGES, GLOBAL_CHANNEL
from seiz_eeg.schemas import ClipsDF
from seiz_eeg.tusz.signals.io import read_parquet
from seiz_eeg.tusz.signals.process import get_diff_signals
from seiz_eeg.utils import make_clips


class EEGDataset(Dataset):
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        segments_df: DataFrame[ClipsDF],
        *,
        clip_length: float,
        clip_stride: Union[int, float, str],
        signal_transform: Optional[Callable] = None,
        diff_channels: Optional[bool] = False,
        node_level: Optional[bool] = False,
        device: Optional[str] = None,
    ) -> None:
        """Dataset of EEG clips with seizure labels

        Args:
            clips_df (DataFrame[ClipsDF]): Pandas dataframe of EEG clips
            node_level (Optional[bool]): Wheter to get node-level or global labels
                (only latter is currently supported)
            device (Optional[str], optional): Torch device. Defaults to None.
        """
        super().__init__()

        logging.debug("Creating clips from segments")
        self.clips_df = make_clips(segments_df, clip_length=clip_length, clip_stride=clip_stride)

        self.diff_channels = diff_channels
        self.node_level(node_level)
        self.device = device

        self.signal_transform = signal_transform

        self.output_shape = self._get_output_shape()

    def node_level(self, node_level: bool):
        """Setter for the node-level labels retrieval"""
        self._node_level = node_level

        if node_level:
            raise NotImplementedError
            # self._clips_df = self.clips_df.drop(GLOBAL_CHANNEL).groupby(AnnotationDF.channel)
        else:
            self._clips_df = self.clips_df.loc[idx[:, :, :, GLOBAL_CHANNEL]]

    def _get_from_df(self, index: int) -> Tuple[Union[int, List[int]], float, float, str]:
        if self._node_level:
            raise NotImplementedError

        return self._clips_df.iloc[index][
            [
                ClipsDF.label,
                ClipsDF.start_time,
                ClipsDF.end_time,
                ClipsDF.sampling_rate,
                ClipsDF.signals_path,
            ]
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, start_time, end_time, s_rate, signals_path = self._get_from_df(index)

        start_sample = int(start_time * s_rate)
        end_sample = int(end_time * s_rate)
        signals = read_parquet(signals_path).iloc[start_sample:end_sample]

        # 1. (opt) Subtract pairwise columns
        if self.diff_channels:
            signals = get_diff_signals(signals, EEG_MONTAGES).values
        else:
            signals = signals.values

        # 2. Convert to torch
        signals = torch.tensor(signals, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)

        if self.signal_transform is not None:
            signals = self.signal_transform(signals)

        return signals, label

    def get_label_array(self) -> np.ndarray:
        return self._clips_df[ClipsDF.label].values

    def get_channels_names(self) -> List[str]:
        if self.diff_channels:
            return EEG_MONTAGES
        else:
            return EEG_CHANNELS

    def compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of signals and store result in ``self.(mean|std)``"""

        # This part is commented for reproducibility, consider reimplementing it if performances are
        # too slow
        # if len(self) > 5000:
        #     rng = default_rng()
        #     samples = rng.choice(len(self), size=5000, replace=False, shuffle=False)
        #     samples.sort()  # We sort the samples to open their files in order, if possible
        # else:
        samples = np.arange(len(self))

        t_sum = torch.zeros(self.output_shape[0], dtype=torch.float64, device=self.device)
        t_sum_sq = torch.zeros_like(t_sum)
        for i in samples:
            X, _ = self[i]
            t_sum += X
            t_sum_sq += X**2

        N = len(samples) * np.prod(self.output_shape[0])
        mean = torch.sum(t_sum) / N
        # Compute std with Bessel's correction
        std = torch.sqrt((torch.sum(t_sum_sq) - N * self.mean**2) / (N - 1))

        return mean, std

    def _get_output_shape(self) -> Tuple[torch.Size, torch.Size]:
        X0, y0 = self.__getitem__(0)
        return X0.shape, y0.shape

    def __len__(self) -> int:
        return len(self._clips_df)

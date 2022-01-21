"""EEG Data class with common data retrieval"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas import IndexSlice as idx
from pandera import check_types
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from src.data.schemas import ClipsDF
from src.data.tusz.constants import CHANNELS, GLOBAL_CHANNEL, MONTAGES
from src.data.tusz.signals.io import read_parquet
from src.data.tusz.signals.process import get_diff_signals


@check_types
def make_clips(
    annotations: DataFrame[ClipsDF],
    clip_length: int,
    clip_stride: int,
) -> DataFrame[ClipsDF]:
    "Split annotations dataframe in dataframe of clips"
    if clip_length < 0:
        return annotations.sort_index()

    index_names = annotations.index.names
    annotations = annotations.reset_index()

    start_times, end_times = (
        annotations[ClipsDF.start_time],
        annotations[ClipsDF.end_time],
    )

    out_list = []
    for clip_idx, clip_start in enumerate(np.arange(0, end_times.max(), clip_stride)):
        clip_end = clip_start + clip_length

        bool_mask = (start_times <= clip_start) & (clip_end <= end_times)

        copy_vals = annotations[bool_mask].copy()
        copy_vals[[ClipsDF.segment, ClipsDF.start_time, ClipsDF.end_time]] = (
            clip_idx,
            clip_start,
            clip_end,
        )
        out_list.append(copy_vals)

    return pd.concat(out_list).set_index(index_names).sort_index()


class EEGDataset(Dataset):
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        segments_df: DataFrame[ClipsDF],
        *,
        clip_length: float,
        clip_stride: float,
        window_len: Optional[int] = -1,
        diff_channels: Optional[bool] = False,
        fft_coeffs: Optional[Tuple[int, int]] = None,
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

        self.clips_df = make_clips(segments_df, clip_length=clip_length, clip_stride=clip_stride)

        self.window_len = window_len
        self.diff_channels = diff_channels
        self.fft_coeffs = fft_coeffs

        self.node_level(node_level)

        self.device = device

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
            signals = get_diff_signals(signals, MONTAGES).values
        else:
            signals = signals.values

        # 2. Convert to torch
        signals = torch.tensor(signals, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)

        # 3. Split windows if asked
        if self.window_len > 0:
            time_axis = 1

            signals = signals.reshape(
                signals.shape[0] // self.window_len,  # nb of windows
                self.window_len * s_rate,  # nb of samples per window (time axis)
                signals.shape[1],  # nb of signals
            )
        else:
            time_axis = 0

        # 3. Compute fft
        if self.fft_coeffs:
            # Define slices to extract
            extractor = len(signals.shape) * [slice(None)]
            extractor[time_axis] = slice(*self.fft_coeffs)

            # Actual fft
            signals = torch.log(torch.abs(torch.fft.rfft(signals, axis=time_axis)))
            signals = signals[extractor]

        return signals, label

    def get_label_array(self):
        return self._clips_df[ClipsDF.label].values

    def get_channels_names(self):
        if self.diff_channels:
            return MONTAGES
        else:
            return CHANNELS

    def __len__(self) -> int:
        return len(self._clips_df)

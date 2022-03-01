"""EEG Data class with common data retrieval"""
import logging
from typing import List, Optional, Tuple, Union

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
        window_len: Optional[int] = -1,
        diff_channels: Optional[bool] = False,
        fft_coeffs: Optional[Tuple[int, int]] = None,
        node_level: Optional[bool] = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
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

        self.window_len = window_len
        self.diff_channels = diff_channels
        self.fft_coeffs = fft_coeffs

        self.node_level(node_level)

        self.device = device

        self.output_shape = self._get_output_shape()

        # Compute signals mean
        if mean is None:
            if std is not None:
                raise ValueError("You passed std but no mean")

            logging.debug("Computing stats (mean and std)")
            self.mean, self.std = self._compute_stats()
        else:
            if std is None:
                raise ValueError("You passed mean but no std")

            logging.debug("Using predefined (mean and std)")
            self.mean = mean
            self.std = std

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

        # 3. Optional: Split windows
        if self.window_len > 0:
            time_axis = 1
            win_len = self.window_len * s_rate

            signals = signals.reshape(
                signals.shape[0] // win_len,  # nb of windows
                win_len,  # nb of samples per window (time axis)
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
            signals = torch.abs(torch.fft.rfft(signals, axis=time_axis))
            signals = signals[extractor]

        # Center data. This is always performed, except for `_compute_mean`
        if hasattr(self, "mean") and self.mean is not None:
            signals -= self.mean

            # Normalize data. This is always performed, except for `_compute_std`
            if hasattr(self, "std") and self.std is not None:
                signals /= self.std

        return signals, label

    def get_label_array(self) -> np.ndarray:
        return self._clips_df[ClipsDF.label].values

    def get_channels_names(self) -> List[str]:
        if self.diff_channels:
            return EEG_MONTAGES
        else:
            return EEG_CHANNELS

    def _compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of signals and store result in ``self.(mean|std)``"""
        self.mean = None
        self.std = None

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
        self.mean = torch.sum(t_sum) / N
        # Compute std with Bessel's correction
        self.std = torch.sqrt((torch.sum(t_sum_sq) - N * self.mean**2) / (N - 1))

        return self.mean, self.std

    def _get_output_shape(self) -> Tuple[torch.Size, torch.Size]:
        X0, y0 = self.__getitem__(0)
        return X0.shape, y0.shape

    def __len__(self) -> int:
        return len(self._clips_df)

"""EEG Data class with common data retrieval"""
from typing import List, Optional, Tuple, Union

import torch
from pandas import IndexSlice as idx
from pandera import check_types
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from src.data.schemas import AnnotationDF
from src.data.tusz.annotations.process import get_channels
from src.data.tusz.constants import GLOBAL_CHANNEL
from src.data.tusz.signals.io import read_eeg_signals
from src.data.tusz.signals.process import process_signals


class EEGDataset(Dataset):
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        clips_df: DataFrame[AnnotationDF],
        *,
        sampling_rate: int,
        window_len: Optional[int] = -1,
        node_level: Optional[bool] = False,
        diff_channels: Optional[bool] = True,
        device: Optional[str] = None,
    ) -> None:
        """Dataset of EEG clips with seizure labels

        Args:
            clips_df (DataFrame[AnnotationDF]): Pandas dataframe of EEG clips
            sampling_rate (int): Samplking rate of EEG signals
            node_level (Optional[bool]): Wheter to get node-level or global labels
                (only latter is currently supported)
            diff_channels (Optional[bool], optional): Wether to retrieve differential signal or raw.
                Defaults to True, i.e. differential.
            device (Optional[str], optional): Torch device. Defaults to None.

        Raises:
            ValueError: If node_level is true and diff_channels is false
        """
        super().__init__()

        self.clips_df = clips_df

        if diff_channels:
            self.diff_labels = get_channels(clips_df).drop(GLOBAL_CHANNEL)
        else:
            if node_level:
                raise ValueError("Diff channels are compulsory when working ad node level")

            self.diff_labels = None

        self.node_level(node_level)

        self.device = device
        self.sampling_rate = sampling_rate
        self.window_len = window_len

    def node_level(self, node_level: bool):
        """Setter for the node-level labels retrieval"""
        self._node_level = node_level

        if node_level:
            self._clips_df = self.clips_df.drop(GLOBAL_CHANNEL).groupby(AnnotationDF.channel)
        else:
            self._clips_df = self.clips_df.loc[idx[:, :, :, GLOBAL_CHANNEL]]

    def _get_from_df(self, index: int) -> Tuple[Union[int, List[int]], float, float, str]:
        if self._node_level:
            raise NotImplementedError

        return self.clips_df.iloc[index][
            [
                AnnotationDF.label,
                AnnotationDF.start_time,
                AnnotationDF.end_time,
                AnnotationDF.edf_path,
            ]
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, start_time, end_time, edf_path = self._get_from_df(index)

        signals, sr_in = read_eeg_signals(edf_path)
        start_sample = int(start_time * sr_in)
        end_sample = int(end_time * sr_in)

        signals = process_signals(
            signals=signals.iloc[start_sample:end_sample],
            sampling_rate_in=sr_in,
            sampling_rate_out=self.sampling_rate,
            window_len=self.window_len,
            diff_labels=self.diff_labels,
        )

        return torch.tensor(signals, device=self.device), torch.tensor(label, device=self.device)

    def __len__(self) -> int:
        return len(self.clips_df)

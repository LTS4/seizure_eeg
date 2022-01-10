"""EEG Data class with common data retrieval"""
from typing import List, Optional, Tuple, Union

import torch
from pandas import IndexSlice as idx
from pandera import check_types
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from src.data.schemas import AnnotationDF
from src.data.tusz.constants import GLOBAL_CHANNEL
from src.data.tusz.signals.io import read_parquet


class EEGDataset(Dataset):
    """Dataset of EEG clips with seizure labels"""

    @check_types
    def __init__(
        self,
        clips_df: DataFrame[AnnotationDF],
        *,
        window_len: Optional[int] = -1,
        node_level: Optional[bool] = False,
        device: Optional[str] = None,
    ) -> None:
        """Dataset of EEG clips with seizure labels

        Args:
            clips_df (DataFrame[AnnotationDF]): Pandas dataframe of EEG clips
            node_level (Optional[bool]): Wheter to get node-level or global labels
                (only latter is currently supported)
            device (Optional[str], optional): Torch device. Defaults to None.
        """
        super().__init__()

        self.clips_df = clips_df

        self.node_level(node_level)

        self.device = device
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

        return self._clips_df.iloc[index][
            [
                AnnotationDF.label,
                AnnotationDF.start_time,
                AnnotationDF.end_time,
                AnnotationDF.sampling_rate,
                AnnotationDF.signals_path,
            ]
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, start_time, end_time, s_rate, signals_path = self._get_from_df(index)

        start_sample = int(start_time * s_rate)
        end_sample = int(end_time * s_rate)
        signals = read_parquet(signals_path).iloc[start_sample:end_sample].values

        # 3. Split windows
        if self.window_len > 0:
            signals = signals.reshape(
                signals.shape[0] // self.window_len,  # nb of windows
                self.window_len,  # nb of samples per window
                signals.shape[1],  # nb of signals
            )

        return torch.tensor(signals, device=self.device), torch.tensor(label, device=self.device)

    def __len__(self) -> int:
        return len(self._clips_df)

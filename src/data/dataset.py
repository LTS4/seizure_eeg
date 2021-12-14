from typing import List, Optional, Tuple, Union

import torch
from pandas import IndexSlice as idx
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from src.data.schemas import AnnotationDF
from src.data.tusz.constants import GLOBAL_CHANNEL
from src.data.tusz.labels.process import get_channels
from src.data.tusz.signals.io import read_eeg_signals
from src.data.tusz.signals.process import process_signals


class EEGDataset(Dataset):
    def __init__(
        self,
        clips_df: DataFrame[AnnotationDF],
        *,
        sampling_rate: int,
        node_level: Optional[bool],
        diff_channels: Optional[bool] = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.clips_df = clips_df

        print(self.clips_df)

        if diff_channels:
            self.diff_labels = get_channels(clips_df).drop(GLOBAL_CHANNEL)
        else:
            if node_level:
                raise ValueError("Diff channels are compulsory when working ad node level")

            self.diff_labels = None

        self.node_level(node_level)

        self.device = device
        self.sampling_rate = sampling_rate

    def node_level(self, node_level: bool):
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

        start_sample = int(start_time * self.sampling_rate)
        end_sample = int(end_time * self.sampling_rate)

        signals = process_signals(
            *read_eeg_signals(edf_path),
            sampling_rate_out=self.sampling_rate,
            diff_labels=self.diff_labels,
        ).values[start_sample:end_sample]

        # if not write_parquet(
        #     signals,
        #     path=output_folder / edf_path.stem / FILE_SIGNALS,
        #     force_rewrite=force_rewrite,
        # ):
        #     nb_existing += 1

        return torch.tensor(label, device=self.device), torch.tensor(signals, device=self.device)

    def __len__(self) -> int:
        return len(self.clips_df)

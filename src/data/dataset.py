from typing import Optional, Tuple

import torch
from pandas.core.indexing import IndexSlice
from pandera.typing import DataFrame
from torch._C import device
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
        diff_channels: Optional[bool] = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.clips_df = clips_df
        self.device = device
        self.sampling_rate = sampling_rate

        if diff_channels:
            self.diff_labels = get_channels(self.clips_df).drop(GLOBAL_CHANNEL)
        else:
            self.diff_labels = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, start_time, end_time, edf_path = self.clips_df.iloc[index][
            [
                AnnotationDF.label,
                AnnotationDF.start_time,
                AnnotationDF.end_time,
                AnnotationDF.edf_path,
            ]
        ]

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

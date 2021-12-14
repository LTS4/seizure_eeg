from typing import Tuple

import torch
from pandas.core.indexing import IndexSlice
from pandera.typing import DataFrame
from torch.utils.data import Dataset

from src.data.schemas import AnnotationDF


class EEGDataset(Dataset):
    def __init__(self, clips_df: DataFrame[AnnotationDF]) -> None:
        super().__init__()

        self.clips_df = clips_df

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label, signal_path = self.clips_df.iloc[index][[AnnotationDF.label, AnnotationDF.edf_path]]
        print(label)
        print(signal_path)
        input()

        # if diff_channels:
        #     diff_labels = get_channels(annotations).drop(GLOBAL_CHANNEL)
        # else:
        #     diff_labels = None

        # signals = process_signals(
        #     *read_eeg_signals(edf_path),
        #     sampling_rate_out=sampling_rate,
        #     diff_labels=diff_labels,
        # )

        # if not write_parquet(
        #     signals,
        #     path=output_folder / edf_path.stem / FILE_SIGNALS,
        #     force_rewrite=force_rewrite,
        # ):
        #     nb_existing += 1
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clips_df)

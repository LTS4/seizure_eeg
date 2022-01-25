"""EEG Data class with common data retrieval"""
from typing import List, Optional, Set, Tuple, Union

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
            self.mean = self._compute_mean()
            self.std = self._compute_std()
        else:
            if std is None:
                raise ValueError("You passed mean but no std")
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
            signals = get_diff_signals(signals, MONTAGES).values
        else:
            signals = signals.values

        # 2. Convert to torch
        signals = torch.tensor(signals, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.long, device=self.device)

        # 3. Optional: Split windows
        if self.window_len > 0:
            time_axis = 1

            signals = signals.reshape(
                signals.shape[0] // self.window_len,  # nb of windows
                self.window_len * s_rate,  # nb of samples per window (time axis)
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
            signals = torch.log(torch.abs(torch.fft.rfft(signals, axis=time_axis)))
            signals = signals[extractor]

        # Center data. This is always performed, except for `get_mean`
        if hasattr(self, "mean") and self.mean is not None:
            signals -= self.mean

        return signals, label

    def get_label_array(self) -> np.ndarray:
        return self._clips_df[ClipsDF.label].values

    def get_channels_names(self) -> List[str]:
        if self.diff_channels:
            return MONTAGES
        else:
            return CHANNELS

    def _compute_mean(self) -> torch.Tensor:
        """Compute mean of signals and store result in ``self.mean``"""
        self.mean = None
        t_sum = torch.zeros(self.output_shape[0], dtype=torch.float64, device=self.device)
        for X, _ in self:
            t_sum += X

        self.mean = torch.sum(t_sum) / (len(self) * np.prod(self.output_shape[0]))

        return self.mean

    def _compute_std(self) -> torch.Tensor:
        """Compute std of signals and store result in ``self.std``

        We must have already computed self.mean and __getitem__ shall return centered signals.
        """
        assert self.mean is not None

        t_sum = torch.zeros(self.output_shape[0], dtype=torch.float64, device=self.device)
        for X, _ in self:
            t_sum += X ** 2

        self.std = torch.sqrt(torch.sum(t_sum) / (len(self) * np.prod(self.output_shape[0]) - 1))

        return self.std

    def _get_output_shape(self) -> Tuple[torch.Size, torch.Size]:
        X0, y0 = self.__getitem__(0)
        return X0.shape, y0.shape

    def __len__(self) -> int:
        return len(self._clips_df)


def _patient_split(segments_df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float) -> Set[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. In that
    case it should be enough to rerun the split procedure.

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe of EEG segments
        ratio_min (float): Minimum fraction of labels to cover
        ratio_max (float): Maximum fraction of labels to cover

    Raises:
        ValueError: If ratio_[min|max] do not satisfy ``0 < ratio_min <= ratio_max < 1``
        AssertionError: If the sampling algorithm fails,
            I.e. it runs out of patients before getting to ratio_min or any patient added surpass
            ratio_max

    Returns:
        Set[str]: Set of selected patients
    """
    if not 0 < ratio_min <= ratio_max < 1:
        raise ValueError("ratio_[min|max] must satisfy ``0 < ratio_min <= ratio_max < 1``")

    labels_by_patient = segments_df.groupby("patient")["label"]
    label_counts = labels_by_patient.value_counts().rename("counts").reset_index(level="label")

    label_counts["counts"] /= label_counts["label"].map(
        label_counts.groupby("label")["counts"].sum()
    )

    # Start with empty selction and no seen patients
    selected = set()
    seen = set()

    # Start by filing up small classes as they have fewer choices
    for label in segments_df["label"].value_counts().sort_values().index:
        # Only look at counts of this instance
        filtered = label_counts[label_counts["label"] == label]

        # We can already have a partial selection
        p_selection = selected & set(filtered.index)
        # Choose from patients representing this class, not already selected and unseen
        to_choose = list(set(filtered.index) - selected - seen)

        # We add elements until we get the desired ratio
        while filtered.loc[p_selection, "counts"].sum() <= ratio_min:
            assert to_choose, "No patients satisfy split, retry"

            # Randomly pick a candidate
            candidate = np.random.choice(to_choose)

            to_choose.remove(candidate)
            p_selection.add(candidate)
            seen.add(candidate)

            # If we pass the theshold the candidate is invalid
            if filtered.loc[p_selection, "counts"].sum() > ratio_max:
                p_selection.discard(candidate)

        selected = selected.union(p_selection)

        seen = seen.union(to_choose)

    return selected


def patient_split(segments_df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float) -> Set[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. This
    functions tries ten different random splits before failing. In that case, with high probability
    no split exists.

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe of EEG segments
        ratio_min (float): Minimum fraction of labels to cover
        ratio_max (float): Maximum fraction of labels to cover

    Raises:
        ValueError: If ratio_[min|max] do not satisfy ``0 < ratio_min <= ratio_max < 1``
        ValueError: If the sampling algorithm fails 10 times, which probably means that
            ratio_[min|max] are too restrictive

    Returns:
        Set[str]: Set of selected patients
    """
    for _ in range(10):
        try:
            selected = _patient_split(segments_df, ratio_min, ratio_max)
            break
        except AssertionError:
            continue
    else:
        raise ValueError("Impossible to create a valid split with given ratio")

    return selected

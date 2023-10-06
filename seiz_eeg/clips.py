"""Functions to extract clips from seizure datasets"""
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandera import check_types
from pandera.typing import DataFrame, Series
from tqdm.autonotebook import tqdm

from seiz_eeg.schemas import ClipsDF


def _get_mask(
    start_times: Series[float],
    end_times: Series[float],
    clip_start: float,
    clip_end: float,
    overlap_action: str,
) -> Series[bool]:
    inner_mask = (start_times <= clip_start) & (clip_end <= end_times)

    if overlap_action == "ignore":
        return inner_mask
    else:
        if overlap_action == "right":
            left_mask = np.zeros_like(inner_mask)
        else:
            # Identify segments which overlap with the clip start
            left_mask = (start_times <= clip_start) & (clip_start < end_times)

        if overlap_action == "left":
            right_mask = np.zeros_like(inner_mask)
        else:
            # Identify segments which overlap with the clip end
            right_mask = (start_times < clip_end) & (clip_end <= end_times)

        if overlap_action not in ("right", "left"):
            outer_mask = (clip_start < start_times) & (end_times < clip_end)
        else:
            outer_mask = np.zeros_like(inner_mask)

        return inner_mask | left_mask | right_mask | outer_mask


def _handle_overlaps(
    copy_vals: DataFrame, index_names: List[str], overlap_action: str
) -> DataFrame:
    """Handle overlapping (duplicated) clips based on specified action.

    Args:
        copy_vals (DataFrame): Intermediate clips dataframe
        index_names (List[str]): List of columns in :attr:`copy_vars` to identify overlaps (indices)
        overlap_action (str): _description_

    Raises:
        AssertionError: _description_

    Returns:
        DataFrame: _description_
    """
    duplicated = copy_vals.duplicated(subset=index_names, keep=False)

    if np.any(duplicated):
        if overlap_action == "seizure":
            # Since background is 0, one seizure label will alway get rtrieved by the max
            # Clip sizes should be small enough to avoid multiple overlaps.
            seiz_labels = copy_vals.loc[duplicated].groupby(index_names)[ClipsDF.label].max()
            copy_vals = copy_vals.set_index(index_names)

            # copy_vals.update(seiz_labels)
            copy_vals.loc[seiz_labels.index, ClipsDF.label] = seiz_labels

            copy_vals = copy_vals.reset_index()
        elif overlap_action == "bkgd":
            copy_vals.loc[duplicated, ClipsDF.label] = 0
        else:
            raise AssertionError(
                f"Duplicated values in clips extraction:\n{copy_vals.loc[duplicated]}"
            )

        copy_vals = copy_vals.drop_duplicates(subset=index_names, keep="first")

    return copy_vals.astype({ClipsDF.label: int})


SUPPORTED_OVERLAP_ACTION = ("ignore", "right", "left", "seizure", "bkgd")


def _make_clips_float_stride(
    segments_df: DataFrame[ClipsDF],
    clip_length: float,
    clip_stride: Union[float, str],
    overlap_action: str = "ignore",
) -> DataFrame[ClipsDF]:
    if clip_stride < 0:
        raise ValueError(f"Clip stride must be postive, got {clip_stride}")

    if overlap_action not in SUPPORTED_OVERLAP_ACTION:
        raise ValueError(
            f"Invalid overlap_action: got {overlap_action}, must be in {SUPPORTED_OVERLAP_ACTION}"
        )

    max_times = segments_df.groupby(level=[ClipsDF.patient, ClipsDF.session]).end_time.max()

    index_names = segments_df.index.names
    segments_df = segments_df.reset_index()

    start_times, end_times = (
        segments_df[ClipsDF.start_time],
        segments_df[ClipsDF.end_time],
    )

    out_list = []
    for clip_idx, clip_start in tqdm(
        enumerate(np.arange(0, end_times.max(), clip_stride)),
        leave=False,
        desc="Clip extraction",
        total=end_times.max() // clip_stride,
    ):
        clip_end = clip_start + clip_length

        copy_vals = segments_df[
            _get_mask(start_times, end_times, clip_start, clip_end, overlap_action)
        ].copy()

        if len(copy_vals) > 0:
            copy_vals[[ClipsDF.segment, ClipsDF.start_time, ClipsDF.end_time]] = (
                clip_idx,
                clip_start,
                clip_end,
            )

            # We only keep the segments finishing before the end of the session
            copy_vals = copy_vals.loc[
                copy_vals.apply(
                    lambda row: row.end_time <= max_times.loc[row.patient, row.session], axis=1
                )
            ]

            out_list.append(_handle_overlaps(copy_vals, index_names, overlap_action))

    return pd.concat(out_list, ignore_index=True, copy=False).set_index(index_names)


def _make_clips_start(
    segments_df: DataFrame[ClipsDF],
    clip_length: float,
) -> DataFrame[ClipsDF]:
    start_times, end_times = (
        segments_df[ClipsDF.start_time],
        segments_df[ClipsDF.end_time],
    )

    clips = segments_df.copy()
    clips[ClipsDF.end_time] = start_times + clip_length  # clips end after given lenght

    # we only keep clips which fall completely in a segment
    return clips.loc[clips[ClipsDF.end_time] <= end_times]


def _make_clips_preictal(
    segments_df: DataFrame[ClipsDF],
    clip_length: float,
    pre_time: Optional[float] = None,
) -> DataFrame[ClipsDF]:
    seiz_clips = segments_df.loc[segments_df[ClipsDF.label] > 0].copy()
    end_times = seiz_clips[ClipsDF.end_time].copy()

    seiz_clips[ClipsDF.end_time] = seiz_clips[ClipsDF.start_time] + clip_length
    seiz_clips = seiz_clips.loc[seiz_clips[ClipsDF.end_time] <= end_times]

    if pre_time is None:
        pre_time = clip_length

    # Extract clips preceding seizures
    bkgd_clips = segments_df.loc[
        # Extract only clips with valid ictal
        seiz_clips.index.set_levels(seiz_clips.index.levels[2] - 1, level="segment")
    ].copy()

    bkgd_clips[ClipsDF.end_time] -= pre_time
    new_starts = bkgd_clips[ClipsDF.end_time] - clip_length
    to_keep = bkgd_clips[ClipsDF.start_time] <= new_starts

    bkgd_clips[ClipsDF.start_time] = new_starts
    bkgd_clips = bkgd_clips[to_keep]

    return pd.concat(
        (
            bkgd_clips,
            seiz_clips.loc[
                # Extract only clips with valid pre-ictal
                bkgd_clips.index.set_levels(bkgd_clips.index.levels[2] + 1, level="segment")
            ],
        )
    )


def _make_clips_random(
    segments_df: DataFrame[ClipsDF],
    clip_length: float,
    seed: Optional[int] = None,
) -> DataFrame[ClipsDF]:

    intervals = segments_df[ClipsDF.end_time] - clip_length - segments_df[ClipsDF.start_time]

    rng = np.random.default_rng(seed)

    rel_start = rng.uniform(size=len(intervals))

    clips = segments_df.copy()
    clips[ClipsDF.start_time] = rel_start * intervals
    clips[ClipsDF.end_time] = clips[ClipsDF.start_time] + clip_length

    # we only keep clips which fall completely in a segment
    return clips.loc[intervals > 0]


@check_types
def make_clips(
    segments_df: DataFrame[ClipsDF],
    clip_length: float,
    clip_stride: Union[float, str, tuple],
    overlap_action: str = "ignore",
    sort_index: bool = False,
) -> DataFrame[ClipsDF]:
    """Split annotations dataframe in dataframe of clips

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe containing annotations for EEG segments
        clip_length (float): Lenght of the output clips, in same
            unit as *start_time* and *end_time* of :attr:`segments_df`. A
            negative value returns the segments unchanged, but sort the dataset
            by index.
        clip_stride (Union[float, str, tuple]): Stride to extract the start times of the clips.
            Integer or real values give explicit stride. If string or tuple,
            must be one of the following, with respective parameters.:
            - ``start``: extract one clip per segment, starting at onset/termination label.
            - ``pre-ictal``: for each onset time extract the beginning of ictal
                segment and the preictal clip ending *clip_lenght*, or specified sec before
                onset time. Only pairs of pre-ictal/ictal clips are returned
            - ``random``: For each segment, extact a random clip of desired length.
        overlap_action (str): What to do with clips overlapping segments.
            Options:
            - ``ignore``: do not include any crossing clips
            - ``left``: the label of crossing clips is given by the left
                (preceding) segment
            - ``right``: the label of crossing clips is given by the right
                (ending) segment
            - ``seizure``: the label of crossing clips is given by the ictal
            segment. (No more than two segments should be crossing)
            - ``bkgd``: set the label of crossing clips to be 0

    Raises:
        ValueError: If ``clip_stride`` is negative, or an invalid string
        ValueError: If ``overlap_action`` is an invalid string

    Returns:
        DataFrame[ClipsDF]: Clips dataframe
    """
    if clip_length < 0:
        clips = segments_df
    else:
        if isinstance(clip_stride, (int, float)):
            clips = _make_clips_float_stride(segments_df, clip_length, clip_stride, overlap_action)
        elif isinstance(clip_stride, str):
            if clip_stride == "start":
                clips = _make_clips_start(segments_df, clip_length)
            elif clip_stride == "pre-ictal":
                clips = _make_clips_preictal(segments_df, clip_length)
            elif clip_stride == "random":
                clips = _make_clips_random(segments_df, clip_length)
            else:
                raise ValueError(f"Invalid clip_stride, got {clip_stride}")
        elif isinstance(clip_stride, tuple):
            stride_name, *args = clip_stride

            if stride_name == "pre-ictal":
                clips = _make_clips_preictal(segments_df, clip_length, *args)
            elif stride_name == "random":
                clips = _make_clips_random(segments_df, clip_length, *args)
            else:
                raise ValueError(f"Invalid key in clip_stride, got {stride_name}")
        else:
            raise ValueError(f"Invalid clip_stride type, got {type(clip_stride)}")

    if sort_index:
        # Sorting indices requires ~70% of the time spent in this function
        # Consider only sorting level=[0, 1]
        return clips.sort_index()
    else:
        return clips

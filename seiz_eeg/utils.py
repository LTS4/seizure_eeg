"""Utility functions for EEG Datasets"""
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas import IndexSlice as idx
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
            left_mask = (clip_start <= start_times) & (start_times <= clip_end)

        if overlap_action == "left":
            right_mask = np.zeros_like(inner_mask)
        else:
            right_mask = (start_times <= clip_end) & (clip_end <= end_times)

        if overlap_action not in ("right", "left"):
            outer_mask = (clip_start <= start_times) & (end_times <= clip_end)
        else:
            outer_mask = np.zeros_like(inner_mask)

        return inner_mask | left_mask | right_mask | outer_mask


def _handle_overlaps(
    copy_vals: DataFrame, index_names: List[str], overlap_action: str
) -> DataFrame:
    """Handle overlapping (duplicated) clips based on specified action.

    Args:
        copy_vals (DataFrame): Intermediate clips dataframe
        index_names (List[str]): List of columns in :var:`copy_vars` to identify overlaps (indices)
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
            copy_vals.update(seiz_labels)
            copy_vals = copy_vals.reset_index()
        elif overlap_action == "bkgd":
            copy_vals.loc[duplicated, ClipsDF.label] = 0
        else:
            raise AssertionError(
                f"Duplicated values in clips extraction:\n{copy_vals.loc[duplicated]}"
            )

        copy_vals = copy_vals.drop_duplicates(subset=index_names, keep="first")

    return copy_vals


def _make_clips_float_stride(
    segments_df: DataFrame[ClipsDF],
    clip_length: Union[int, float],
    clip_stride: Union[int, float, str],
    overlap_action: str = "ignore",
) -> DataFrame[ClipsDF]:
    if clip_stride < 0:
        raise ValueError(f"Clip stride must be postive, got {clip_stride}")

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

        copy_vals[[ClipsDF.segment, ClipsDF.start_time, ClipsDF.end_time]] = (
            clip_idx,
            clip_start,
            clip_end,
        )

        out_list.append(_handle_overlaps(copy_vals, index_names, overlap_action))

    return pd.concat(out_list, ignore_index=True, copy=False).set_index(index_names)


def _make_clips_start(
    segments_df: DataFrame[ClipsDF],
    clip_length: Union[int, float],
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
    clip_length: Union[int, float],
) -> DataFrame[ClipsDF]:
    seiz_clips = segments_df.loc[segments_df[ClipsDF.label] > 0].copy()
    end_times = seiz_clips[ClipsDF.end_time].copy()

    seiz_clips[ClipsDF.end_time] = seiz_clips[ClipsDF.start_time] + clip_length
    seiz_clips = seiz_clips.loc[seiz_clips[ClipsDF.end_time] <= end_times]

    # Extract clips preceding seizures
    bkgd_clips = segments_df.loc[
        # Extract only clips with valid ictal
        seiz_clips.index.set_levels(seiz_clips.index.levels[2] - 1, level="segment")
    ].copy()

    bkgd_clips[ClipsDF.end_time] -= clip_length
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


@check_types
def make_clips(
    segments_df: DataFrame[ClipsDF],
    clip_length: Union[int, float],
    clip_stride: Union[int, float, str],
    overlap_action: str = "ignore",
) -> DataFrame[ClipsDF]:
    """Split annotations dataframe in dataframe of clips

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe containing annotations for EEG segments
        clip_length (Union[int, float]): Lenght of the output clips, in same
            unit as *start_time* and *end_time* of :arg:`segments_df`. A
            negative value returns the segments unchanged, but sort the dataset
            by index.
        clip_stride (Union[int, float, str]): Stride to extract the start times of the clips.
            Integer or real values give explicit stride. If string, must be one of the following:
                - ``start``: extract one clip per segment, starting at onset/termination label.
                - ``pre-ictal``: for each onset time extract the beginning of ictal
                    segment and the preictal clip ending *clip_lenght* sec before onset time.
                    Only pair of pre-ictal/ictal clips are returned
        overlap_action (str): What to do with clips overlapping segments.
            Options:
                - ``ignore``: do not include any crossing clips
                - ``left``:the label of crossing clips is given by the left
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
        return segments_df.sort_index()

    if isinstance(clip_stride, (int, float)):
        clips = _make_clips_float_stride(segments_df, clip_length, clip_stride, overlap_action)
    elif clip_stride == "start":
        clips = _make_clips_start(segments_df, clip_length)

    elif clip_stride == "pre-ictal":
        clips = _make_clips_preictal(segments_df, clip_length)
    else:
        raise ValueError(f"Invalid clip_stride, got {clip_stride}")

    # Sorting indices requires ~70% of the time spent in this function
    return clips.sort_index()


def _patient_split(
    segments_df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float, rng: np.random.Generator
) -> List[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. In that
    case it should be enough to rerun the split procedure.

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe of EEG segments
        ratio_min (float): Minimum fraction of labels to cover
        ratio_max (float): Maximum fraction of labels to cover
        rng (np.random.Generator): Random number generator

    Raises:
        ValueError: If ratio_[min|max] do not satisfy ``0 < ratio_min <= ratio_max < 1``
        AssertionError: If the sampling algorithm fails,
            I.e. it runs out of patients before getting to ratio_min or any patient added surpass
            ratio_max

    Returns:
        List[str]: List of selected patients
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
        p_selection = list(selected & set(filtered.index))
        # Choose from patients representing this class, not already selected and unseen
        to_choose = list(set(filtered.index) - selected - seen)
        # We need to sort patients for reproducibility as sets have non-deterministic ordering.
        to_choose.sort()

        # We add elements until we get the desired ratio
        while filtered.loc[p_selection, "counts"].sum() <= ratio_min:
            assert to_choose, "No patients satisfy split, retry"

            # Randomly pick a candidate
            candidate = rng.choice(to_choose)
            seen.add(candidate)
            to_choose.remove(candidate)
            p_selection.append(candidate)

            # If we pass the theshold the candidate is invalid
            if filtered.loc[p_selection, "counts"].sum() > ratio_max:
                p_selection.remove(candidate)

        selected = selected.union(p_selection)

        seen = seen.union(to_choose)

    # Sort the list for reproducibility
    selected = list(selected)
    selected.sort()

    ratios = label_counts.loc[selected].groupby("label").sum()
    logging.debug("Splitted with ratios %s", ratios.to_dict())
    return selected


def patient_split(
    segments_df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float, seed: Optional[int] = None
) -> List[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. This
    functions tries ten different random subsets before failing. In that case, with high probability
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
        List[str]: List of selected patients
    """
    rng = default_rng(seed)

    for _ in range(10):
        try:
            selected = _patient_split(segments_df, ratio_min, ratio_max, rng=rng)
            break
        except AssertionError:
            continue
    else:
        raise ValueError("Impossible to create a valid split with given ratio")

    return selected


def extract_by_seizures(segments_df: DataFrame[ClipsDF], min_nb_seiz: int) -> DataFrame[ClipsDF]:
    """Extract only sessions with at least :var:`min_nb_seiz`.

    Args:
        segments_df (DataFrame[ClipsDF]): Segments annotation dataframe
        min_nb_seiz (int): Minumum number of seizures per session (inclusive)

    Returns:
        DataFrame[ClipsDF]: Annotation dataframe with only requested sessions
    """

    gsegments_df = segments_df.loc[idx[:, :, :, "global"]]
    sizes = gsegments_df[gsegments_df[ClipsDF.label] > 0].groupby("session").size()
    to_keep = sizes[sizes >= min_nb_seiz].index

    return segments_df.loc[idx[:, to_keep, :, :]]

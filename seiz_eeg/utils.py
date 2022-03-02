"""Utility functions for EEG Datasets"""
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandera import check_types
from pandera.typing import DataFrame

from seiz_eeg.schemas import ClipsDF


@check_types
def make_clips(
    annotations: DataFrame[ClipsDF],
    clip_length: Union[int, float],
    clip_stride: Union[int, float, str],
) -> DataFrame[ClipsDF]:
    """Split annotations dataframe in dataframe of clips

    Args:
        annotations (DataFrame[ClipsDF]): Dataframe containing seizure annotations
        clip_length (Union[int, float]): Lenght of the output clips, in same unit as ``start_time``
            and ``end_time`` of *annotations*. A negative value returns the
            segments unchanged, but sort the dataset by index.
        clip_stride (Union[int, float, str]): Stride to extract the start times of the clips.
            Integer or real values give explicit stride. If string, must be one of the following:
                - "start": extract one clip per segment, starting at onset/termination label.

    Raises:
        ValueError: If ``clip_stride`` is negative, or an invalid string

    Returns:
        DataFrame[ClipsDF]: Clips dataframe
    """
    if clip_length < 0:
        return annotations.sort_index()

    index_names = annotations.index.names
    annotations = annotations.reset_index()

    start_times, end_times = (
        annotations[ClipsDF.start_time],
        annotations[ClipsDF.end_time],
    )

    if isinstance(clip_stride, (int, float)):
        if clip_stride < 0:
            raise ValueError(f"Clip stride must be postive, got {clip_stride}")

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

        clips = pd.concat(out_list)
    elif clip_stride == "start":
        clips = annotations.copy()
        clips[ClipsDF.end_time] = start_times + clip_length  # Clips end after given lenght

        # We only keep clips which fall completely in a segment
        # TODO: Consider wether changing this
        clips = clips.loc[clips[ClipsDF.end_time] <= end_times]
    else:
        raise ValueError(f"Invalid clip_stride, got {clip_stride}")

    return clips.set_index(index_names).sort_index()


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

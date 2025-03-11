"""Utility functions for EEG Datasets"""

import logging
from typing import List, Optional
from warnings import warn

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas import IndexSlice as idx
from pandera.typing import DataFrame

from seiz_eeg.schemas import ClipsDF


def cut_long_sessions(segments_df: DataFrame[ClipsDF], max_time: float) -> DataFrame[ClipsDF]:
    """Cut EEG session longer than :attr:`max_time`.

    If :attr:`max_time` is smaller than zero then :attr:`segments_df` is returned unchanged.

    Args:
        segments_df (DataFrame[ClipsDF]): Dataframe of EEG segments
        max_time (float): Cutoff time

    Returns:
        DataFrame[ClipsDF]: Datset of clipped sessions.
    """
    if max_time > 0:
        segments_df = segments_df.loc[segments_df[ClipsDF.start_time] < max_time].copy()
        segments_df.loc[segments_df[ClipsDF.end_time] >= max_time, ClipsDF.end_time] = max_time

    return segments_df


def _patient_split(
    df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float, rng: np.random.Generator
) -> List[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. In that
    case it should be enough to rerun the split procedure.

    Args:
        df (DataFrame[ClipsDF]): Dataframe of EEG segments
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

    labels_by_patient = df.groupby("patient")["label"]
    label_counts = labels_by_patient.value_counts().rename("counts").reset_index(level="label")

    label_counts["counts"] /= label_counts["label"].map(
        label_counts.groupby("label")["counts"].sum()
    )

    # Start with empty selction and no seen patients
    selected = set()
    seen = set()

    # Start by filing up small classes as they have fewer choices
    for label in df["label"].value_counts().sort_values().index:
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
    df: DataFrame[ClipsDF], ratio_min: float, ratio_max: float, seed: Optional[int] = None
) -> List[str]:
    """Compute a set of patients from segments_df indices such that they represent between ratio min
    and max of each label appearences.

    Patients are randomly sampled to satisy the constraint on each label iteratively. In some cases,
    previous selections make it impossible to satisfy the constraint by adding any patient. This
    functions tries ten different random subsets before failing. In that case, with high probability
    no split exists.

    Args:
        df (DataFrame[ClipsDF]): Dataframe of EEG segments
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
            selected = _patient_split(df, ratio_min, ratio_max, rng=rng)
            break
        except AssertionError:
            continue
    else:
        raise ValueError("Impossible to create a valid split with given ratio")

    return selected


################################################################################
# NUMBER OF SEIZURES ###########################################################


def patients_by_seizures(
    segments_df: DataFrame[ClipsDF], low: int = 0, high: int = np.inf, total=False
) -> DataFrame[ClipsDF]:
    """Filter patients to have a number of seizures between low and high

    Args:
        segments_df (DataFrame[ClipsDF]): Segments annotation dataframe
        min_nb_seiz (int, optinal): Minumum number of seizures per patient
            (inclusive). Defaults to 0.
        max_nb_seiz (int, optinal): Maximum number of seizures per patient
            (inclusive). Defaults to infinity.
        total (bool, optional): Wether to aggregate all kind of seizures in the
            count. Defaults to False.

    Returns:
        DataFrame[ClipsDF]: Filtered dataframe
    """
    if low <= 0 and high == np.inf:
        return segments_df
    if high < low:
        raise ValueError(f"Low bound must be greater or equal to high, got {high} < {low}")

    if total:
        counts = segments_df.loc[segments_df.label.ne(0)].groupby(ClipsDF.patient).size()
        return segments_df.loc[counts.index[counts.between(low, high)]].copy()
    else:
        counts = (
            segments_df.loc[segments_df.label.ne(0)]
            .groupby(ClipsDF.patient)[ClipsDF.label]
            .value_counts()
        )

        return segments_df.loc[
            counts.index[counts.between(low, high)].get_level_values(ClipsDF.patient).unique()
        ].copy()


def sessions_by_seizures(
    segments_df: DataFrame[ClipsDF], low: int = 0, high: int = np.inf
) -> DataFrame[ClipsDF]:
    """Extract only sessions with at least :attr:`min_nb_seiz`.

    Args:
        segments_df (DataFrame[ClipsDF]): Segments annotation dataframe
        min_nb_seiz (int): Minumum number of seizures per session (inclusive)

    Returns:
        DataFrame[ClipsDF]: Annotation dataframe with only requested sessions
    """
    if low <= 0 and high == np.inf:
        return segments_df
    if high < low:
        raise ValueError(f"Low bound must be greater or equal to high, got {high} < {low}")

    if low == 0:
        # Keep sessions with no seizures, i.e. with only one bkgd segment
        num_segments = (
            segments_df.reset_index(level=ClipsDF.segment)[ClipsDF.segment]
            .groupby(ClipsDF.session)
            .max()
        )
        sess_to_keep = num_segments[num_segments == 0].index.to_list()
    else:
        sess_to_keep = []

    num_seiz = (
        segments_df[segments_df[ClipsDF.label] > 0]
        .groupby(level=[ClipsDF.patient, ClipsDF.session])
        .size()
    )
    sess_to_keep += num_seiz[num_seiz.between(low, high)].index.to_list()
    # sess_to_keep = np.array(sess_to_keep)

    return (
        segments_df.reset_index(level=ClipsDF.segment)
        .loc[sess_to_keep]
        .set_index(ClipsDF.segment, append=True)
    )


################################################################################
# LABELS FILTERING #############################################################


def extract_target_labels(
    df: DataFrame[ClipsDF], target_labels: List[int], relabel: bool = False
) -> DataFrame[ClipsDF]:
    """Old name of :func:`segments_by_labels`. DEPRECATED"""
    warn(
        "`extract_target_labels` is deprecated and will be removed"
        ", use `segments_by_labels` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return segments_by_labels(df, target_labels, relabel)


def _relabel(df: DataFrame[ClipsDF], target_labels: List[int]) -> DataFrame[ClipsDF]:
    lmap = {label: i for i, label in enumerate(target_labels)}
    df["label"] = df["label"].map(lmap)
    return df


def segments_by_labels(
    df: DataFrame[ClipsDF], target_labels: List[int], relabel: bool = False
) -> DataFrame[ClipsDF]:
    """Extract rows of :attr:`df` whose labels are in :attr:`target_labels`

    Args:
        df (DataFrame[ClipsDF]): Dataframe of EEG clips
        target_labels (List[int]): List of integer labels to extract
        relabel(bool): Whether to relabel `target_labels` progressively from 0

    Returns:
        DataFrame[ClipsDF]: Subset of :attr:`df` with desired labels.
    """

    df = df.loc[df["label"].isin(target_labels)].copy()
    if relabel:
        return _relabel(df, target_labels)
    return df


def sessions_by_labels(
    df: DataFrame[ClipsDF], target_labels: List[int], relabel: bool = False
) -> DataFrame[ClipsDF]:
    """Extract sessions of :attr:`df` whose labels are in :attr:`target_labels`

    Args:
        df (DataFrame[ClipsDF]): Dataframe of EEG clips
        target_labels (List[int]): List of integer labels to extract
        relabel(bool): Whether to relabel `target_labels` progressively from 0

    Returns:
        DataFrame[ClipsDF]: Subset of :attr:`df` with desired labels.
    """
    label_sets = df.groupby(ClipsDF.session)[ClipsDF.label].apply(set)
    sessions = label_sets.index[label_sets <= set(target_labels)]
    df = df.loc[idx[:, sessions, :]]
    if relabel:
        return _relabel(df, target_labels)
    return df


def resample_label(
    df: DataFrame[ClipsDF], label: int, ratio: float = 1, seed: Optional[int] = None
) -> DataFrame[ClipsDF]:
    """_summary_

    Args:
        df (DataFrame[ClipsDF]): Dataframe of EEG clips or segments
        label (int): Label to resample.
        ratio(float, optional): Ratio of desired samples w.r.t. the total count
            of other labels. If the desired :attr:`ratio` exceeds the label
            counts, then the label is bootstrapped (sampled with replacement),
            otherwise it is downsampled (no replacement). Defaults to 1.
        seed (Optional[int], optional): Random seed. Defaults to None.

    Returns:
        DataFrame[ClipsDF]: Dataframe with target class resampled.
    """

    target_mask = df[ClipsDF.label] == label
    target_idx = df.index[target_mask]
    other_idx = df.index[~target_mask]

    nb_resampled = int(ratio * len(other_idx))

    rng = default_rng(seed)
    target_idx = pd.MultiIndex.from_tuples(
        rng.choice(target_idx, nb_resampled, replace=nb_resampled > len(target_idx), shuffle=False)
    )

    return df.loc[other_idx.append(target_idx)].copy()

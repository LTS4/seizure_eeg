"""Utilities to read edf annotations.

The main purpose is to produce a dataset whose entries have this structure:

======= ======= ======= =======  ===== ========== ======== ========= =============
Multiindex                       Columns
-------------------------------  -------------------------------------------------
patient session channel segment  label start_time end_time file_path sampling_rate
======= ======= ======= =======  ===== ========== ======== ========= =============
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pandera import check_types
from pandera.typing import DataFrame, Index

from src.data.schemas import AnnotationDF, LabelDF
from src.data.tusz.annotations.io import read_labels
from src.data.tusz.utils import extract_session_date


def get_channels(annotations: DataFrame[AnnotationDF]) -> Index[str]:
    return annotations.index.get_level_values("channel").unique()


@check_types
def make_clips(
    annotations: DataFrame[AnnotationDF],
    clip_length: int,
    clip_stride: int,
) -> DataFrame[AnnotationDF]:
    "Split annotations dataframe in dataframe of clips"
    if clip_length < 0:
        return annotations.sort_index()

    index_names = annotations.index.names
    annotations = annotations.reset_index()

    start_times, end_times = (
        annotations[AnnotationDF.start_time],
        annotations[AnnotationDF.end_time],
    )

    out_list = []
    for clip_idx, clip_start in enumerate(np.arange(0, end_times.max(), clip_stride)):
        clip_end = clip_start + clip_length

        bool_mask = (start_times <= clip_start) & (clip_end <= end_times)

        copy_vals = annotations[bool_mask].copy()
        copy_vals[[AnnotationDF.segment, AnnotationDF.start_time, AnnotationDF.end_time]] = (
            clip_idx,
            clip_start,
            clip_end,
        )
        out_list.append(copy_vals)

    return pd.concat(out_list).set_index(index_names).sort_index()


def labels_to_annotations(
    df: DataFrame[LabelDF],
    edf_path: Path,
    signals_path: Path,
    sampling_rate: int,
) -> DataFrame[AnnotationDF]:
    """Add [patient, session, date, sampling_rate, signals_path] columns.
    patient, session, and date are extrapolated from edf_path"""
    df[AnnotationDF.patient] = edf_path.parents[1].stem
    df[AnnotationDF.session] = edf_path.stem
    df[AnnotationDF.date] = extract_session_date(edf_path.parents[0].stem)

    df[AnnotationDF.sampling_rate] = sampling_rate
    df[AnnotationDF.signals_path] = str(signals_path.absolute())
    return df


def map_labels(df: DataFrame, label_map: Dict[str, int]) -> DataFrame:
    """Map labels using dictioanry and convert column"""
    n_in = len(df)
    df[AnnotationDF.label] = df[AnnotationDF.label].map(lambda x: label_map[x])
    df = df.dropna().astype({AnnotationDF.label: int})
    logging.debug("Dropped %d entries with nan values", n_in)

    return df


@check_types
def process_annotations(
    edf_path: Path,
    *,
    label_map: Dict[str, int],
    binary: bool,
    signals_path: Path,
    sampling_rate: int,
) -> DataFrame[AnnotationDF]:
    """Read annotations files associated to *edf_path* and add metadata

    Args:
        edf_path (Path): [description]
        label_map (Dict[str, int]): [description]
        binary (bool): [description]
        signals_path (Path): [description]
        sampling_rate (int): [description]

    Returns:
        [type]: [description]
    """
    return (
        read_labels(edf_path, binary=binary)
        .pipe(labels_to_annotations, edf_path, signals_path, sampling_rate)
        .pipe(map_labels, label_map=label_map)
        .set_index(
            [
                AnnotationDF.patient,
                AnnotationDF.session,
                AnnotationDF.segment,
                AnnotationDF.channel,
            ]
        )
    )


####################################################################################################


def time_to_samples(annotations: DataFrame[AnnotationDF], nb_samples) -> DataFrame[AnnotationDF]:
    """Convert the start/end time columns of *annotations* to sample indices based on total
    ``nb_samples``"""
    duration = annotations[AnnotationDF.end_time].max()

    annotations["start_sample"] = np.floor(
        annotations[AnnotationDF.start_time] / duration * nb_samples
    ).astype(int)
    annotations["end_sample"] = np.ceil(
        annotations[AnnotationDF.end_time] / duration * nb_samples
    ).astype(int)

    return annotations


def make_label_masks(
    annotations: DataFrame[AnnotationDF],
    nb_samples: int,
    seiz_voc: Dict[str, int],
) -> DataFrame:
    """DEPRECATED: Create integer encoded target matrix for multiclass classification."""
    duration = annotations.groupby(level="channel")["end_time"].max().unique().item()
    channels = get_channels(annotations)

    mask = pd.DataFrame(np.zeros((nb_samples, len(channels)), dtype=int), columns=channels)

    for (_, _, channel, _), label, start, end, _ in annotations.itertuples():
        # We choose to always use the floor as low and ceil as high
        low = int(np.floor(start / duration * nb_samples))
        high = int(np.ceil(end / duration * nb_samples))

        mask.loc[low:high, channel] = seiz_voc[label]

    return mask

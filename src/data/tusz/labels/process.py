"""Utilities to read edf annotations.

The main purpose is to produce a dataset whose entries have this structure:

======= ======= ======= =======  ===== ========== ======== ========= =============
Multiindex                       Columns
-------------------------------  -------------------------------------------------
patient session channel segment  label start_time end_time file_path sampling_rate
======= ======= ======= =======  ===== ========== ======== ========= =============
"""

from typing import Dict

import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Index

from src.data.schemas import AnnotationDF


def get_channels(annotations: DataFrame[AnnotationDF]) -> Index[str]:
    return annotations.index.get_level_values("channel").unique()


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

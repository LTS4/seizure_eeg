# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import click
import numpy as np
import pandas as pd
from pandera import check_types
from pandera.typing import DataFrame, Index
from tqdm import tqdm

from ...config import TUSZ_VERSION, Signals
from ...run import run
from ..schemas import AnnotationDF
from .annotations import get_edf_annotations
from .constants import (
    FILE_SEGMENTS_DF,
    FILE_SIGNAL_DIFF,
    FILE_SIGNAL_REF,
    GLOBAL_CHANNEL,
    TEMPLATE_SIGNAL_CHANNELS,
)
from .io import read_seiz_vocabulary
from .signals import extract_segment, get_resampled_signals
from .utils import list_all_edf_files

logger = logging.getLogger(__name__)

################################################################################
# DATASET


def get_channels(annotations: DataFrame[AnnotationDF]) -> Index[str]:
    return annotations.index.get_level_values("channel").unique()


def time_to_samples(annotations: DataFrame[AnnotationDF], nb_samples) -> DataFrame[AnnotationDF]:
    """Convert the start/end time columns of *annotations* to sample indices based on total ``nb_samples``"""
    duration = annotations[AnnotationDF.end_time].max()

    annotations["start_sample"] = np.floor(annotations[AnnotationDF.start_time] / duration * nb_samples).astype(int)
    annotations["end_sample"] = np.ceil(annotations[AnnotationDF.end_time] / duration * nb_samples).astype(int)

    return annotations


@check_types
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

    # Save labels to file
    # labels_filepath = session_folder / "labels.parquet"
    # if not force_rewrite and labels_filepath.exists():
    #     logger.info("Skipping labels since file exists: %s", labels_filepath)
    #     nb_existing += 1

    #     label_channels = pd.read_parquet(labels_filepath).columns.drop(GLOBAL_CHANNEL)
    # else:
    # labels = make_label_masks(
    #     annotations,
    #     nb_samples=len(signals),
    #     seiz_voc=seizure_voc,
    # )
    # labels.to_parquet(labels_filepath)

    # label_channels = labels.columns.drop(GLOBAL_CHANNEL)


def process_dataset(
    root_folder: Path,
    output_folder: Path,
    *,
    seizure_voc: Dict[str, int],
    sampling_rate: int,
    binary: bool,
    force_rewrite: Optional[bool] = False,
):
    """Precess every file in the root_folder tree"""
    file_list = list_all_edf_files(root_folder)

    nb_errors_skipped = 0
    nb_existing = 0

    meta = dict(
        sampling_rate=sampling_rate,
        file_signal_ref=FILE_SIGNAL_REF,
        file_signal_diff=FILE_SIGNAL_DIFF,
    )

    annotations_list = []

    for edf_path in tqdm(file_list, desc="Processing dataset"):
        try:
            session_id = edf_path.stem
            session_folder = output_folder / session_id
            os.makedirs(session_folder, exist_ok=True)

            signals = get_resampled_signals(edf_path, sampling_rate)

            # fmt: off
            annotations = (
                get_edf_annotations(edf_path, binary)
                .pipe(time_to_samples, len(signals))
            )
            # fmt: on

            annotations["session_folder"] = str(session_folder)

            annotations_list.append(annotations)

            # Define signals filenames
            signal_ref_path = session_folder / FILE_SIGNAL_REF
            signal_diff_path = session_folder / FILE_SIGNAL_DIFF

            label_channels = get_channels(annotations).drop(GLOBAL_CHANNEL)

            # Save signals to files
            if not force_rewrite and signal_ref_path.exists():
                logger.info("Skipping raw signals since file exists: %s", signal_ref_path)
                nb_existing += 1
            else:
                # TODO: check wheter saving pre-splitted files speeds-up computations
                signals.to_parquet(signal_ref_path)

            if not force_rewrite and signal_diff_path.exists():
                logger.info("Skipping differential signals since file exists: %s", signal_diff_path)
                nb_existing += 1
            else:
                try:
                    loc_signals = pd.DataFrame(np.empty((len(signals), len(label_channels))), columns=label_channels)

                    for diff_label in label_channels:
                        el1, el2 = diff_label.split("-")
                        loc_signals[diff_label] = (
                            signals[TEMPLATE_SIGNAL_CHANNELS.format(el1)]
                            - signals[TEMPLATE_SIGNAL_CHANNELS.format(el2)]
                        )

                    loc_signals.to_parquet(signal_diff_path)
                except KeyError as e:
                    raise KeyError(
                        f"KeyError in session '{session_id}' with signal columns: {signals.columns.tolist()}"
                    ) from e

        except (IOError, AssertionError) as err:
            logger.info("Skipping file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err)
            nb_errors_skipped += 1

    # Create segments database and save
    pd.concat(annotations_list, ignore_index=False).to_parquet(
        session_folder / FILE_SEGMENTS_DF,
    )

    if nb_errors_skipped:
        logger.warning("Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped)
    if nb_existing:
        logger.warning("Skipped %d files which existed already, set level to INFO for details", nb_existing)


################################################################################
# MAIN


@click.command()
@click.argument("raw-data-folder", type=click.Path(exists=True, path_type=Path))
@click.argument("processed-data-folder", type=click.Path(path_type=Path))
@click.option("--force-rewrite", is_flag=True)
def main(raw_data_folder: Path, processed_data_folder: Path, force_rewrite: bool):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("making final data set from raw data")

    seiz_voc = read_seiz_vocabulary(raw_data_folder / TUSZ_VERSION / "_DOCS/seizures_types_v02.xlsx")

    raw_edf_folder = raw_data_folder / TUSZ_VERSION / "edf/dev"
    output_folder = processed_data_folder / TUSZ_VERSION / "dev"

    process_dataset(
        raw_edf_folder,
        output_folder,
        seizure_voc=seiz_voc.set_index("label_str")["label_int"],
        sampling_rate=Signals.sampling_rate,
        binary=False,
        force_rewrite=force_rewrite,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run(main, "tusz/make_dataset.log")

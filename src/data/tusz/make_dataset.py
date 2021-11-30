# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path
from typing import Dict

import click
import numpy as np
import pandas as pd
from pandera import check_types
from pandera.typing import DataFrame
from tqdm import tqdm

from ...config import TUSZ_VERSION, Signals
from ...run import run
from ..schemas import AnnotationSchema
from .annotations import get_edf_annotations
from .io import read_seiz_vocabulary
from .signals import extract_segment, get_sampled_signals_and_names
from .utils import list_all_edf_files

logger = logging.getLogger(__name__)

################################################################################
# DATASET


@check_types
def make_label_masks(
    annotations: DataFrame[AnnotationSchema],
    nb_samples: int,
    seiz_voc: Dict[str, int],
) -> DataFrame:
    """Create integer encoded target matrix for multiclass classification"""

    duration = annotations.groupby(level="channel")["end_time"].max().unique().item()

    channels = annotations.index.get_level_values("channel").unique()

    mask = pd.DataFrame(np.zeros((nb_samples, len(channels)), dtype=int), columns=channels)

    for (_, _, channel, _), label, start, end, _ in annotations.itertuples():
        # We choose to always use the floor as low and ceil as high
        low = int(np.floor(start / duration * nb_samples))
        high = int(np.ceil(end / duration * nb_samples))

        mask.loc[low:high, channel] = seiz_voc[label]

    return mask


def process_dataset(
    root_folder: Path, output_folder: Path, *, seizure_voc: Dict[str, int], sampling_rate: int, binary: bool
):
    """Precess every file in the root_folder tree"""
    file_list = list_all_edf_files(root_folder)

    labels_folder = output_folder / "labels"
    os.makedirs(labels_folder, exist_ok=True)

    nb_skipped = 0
    nb_existing = 0

    for edf_path in tqdm(file_list, desc="Processing dataset"):
        try:
            signals, signal_channels = get_sampled_signals_and_names(edf_path, sampling_rate)

            # Save labels to file
            label_filepath = (labels_folder / edf_path.stem).with_suffix(".parquet")
            if label_filepath.exists():
                logger.info("Skipping labels since file exists: %s", label_filepath)
                nb_skipped += 1
            else:
                annotations = get_edf_annotations(edf_path, binary)

                labels = make_label_masks(
                    annotations,
                    nb_samples=signals.shape[1],
                    seiz_voc=seizure_voc,
                )
                labels.to_parquet()

        except (IOError, AssertionError) as err:
            logger.info("Skipping file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err)
            nb_skipped += 1

    if nb_skipped:
        logger.warning("Skipped %d files raising errors, set level to INFO for details", nb_skipped)
    if nb_existing:
        logger.warning("Skipped %d files which existed already, set level to INFO for details", nb_existing)


################################################################################
# MAIN


@click.command()
@click.argument("raw-data-folder", type=click.Path(exists=True, path_type=Path))
@click.argument("processed-data-folder", type=click.Path(path_type=Path))
def main(raw_data_folder: Path, processed_data_folder: Path):
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
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run(main, "tusz/make_dataset.log")

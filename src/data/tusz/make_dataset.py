# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path
from typing import Dict

import click
from pandas import IndexSlice as idx
from tqdm import tqdm

from ...config import TUSZ_VERSION, Signals
from ...run import run
from .annotations import get_edf_annotations
from .io import read_seiz_vocabulary
from .signals import extract_segment, get_sampled_signals_and_names
from .utils import list_all_edf_files

logger = logging.getLogger(__name__)

################################################################################
# DATASET


def process_file(edf_path: Path, seiz_voc: Dict[str, int], sampling_rate: int, *, binary: bool):
    """Read annotations and signals, then create label masks for signals and save them"""
    annotations = get_edf_annotations(edf_path, binary)
    signals, signal_channels = get_sampled_signals_and_names(edf_path, sampling_rate)

    # General labels
    # annotations_general = annotations.loc[idx[:, :, "general", :]]

    # for index, label, start_time, end_time, _ in annotations.itertuples(name=None):
    #     print(index, label, start_time, end_time)


def process_dataset(root_folder: Path, *, seizure_voc: Dict[str, int], sampling_rate: int, binary: bool):
    """Precess every file in the root_folder tree"""
    file_list = list_all_edf_files(root_folder)

    nb_skipped = 0

    for file_path in tqdm(file_list, desc="Processing dataset"):
        try:
            process_file(file_path, seizure_voc, sampling_rate=sampling_rate, binary=binary)
        except (IOError, AssertionError) as err:
            logger.info("Skipping file %s wich raises %s: \n\t%s", file_path, type(err).__name__, err)
            nb_skipped += 1

    if nb_skipped:
        logger.warning("Skipped %d files raising errors, set level to INFO for details", nb_skipped)


################################################################################
# MAIN


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True, path_type=Path))
@click.argument("output_filepath", type=click.Path(path_type=Path))
def main(input_filepath: Path, output_filepath: Path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("making final data set from raw data")

    os.makedirs(output_filepath, exist_ok=True)

    seiz_voc = read_seiz_vocabulary(input_filepath / TUSZ_VERSION / "_DOCS/seizures_types_v02.xlsx")

    process_dataset(
        Path(input_filepath) / TUSZ_VERSION / "edf/dev",
        seizure_voc=seiz_voc,
        sampling_rate=Signals.sampling_rate,
        binary=False,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run(main, "tusz/make_dataset.log")

# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
from pathlib import Path

import click

from src.config import TUSZ, Signals
from src.data.tusz.dataset import process_dataset
from src.run import run

logger = logging.getLogger(__name__)


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

    raw_edf_folder = raw_data_folder / TUSZ.version / "edf"
    output_folder = processed_data_folder / TUSZ.version

    for split in ("dev", "train"):
        process_dataset(
            raw_edf_folder / split,
            output_folder / split,
            sampling_rate=Signals.sampling_rate,
            diff_channels=Signals.diff_channels,
            binary=False,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run(main, "data/tusz.log")

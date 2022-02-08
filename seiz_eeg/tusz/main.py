# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

from seiz_eeg.config import DataConf
from seiz_eeg.tusz.download import download
from seiz_eeg.tusz.io import write_parquet
from seiz_eeg.tusz.process import process_walk

################################################################################
# MAIN


def main(cfg: DataConf):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    load_dotenv(find_dotenv())

    raw_edf_folder = Path(cfg.tusz.raw_edf)
    output_folder = Path(cfg.tusz.processed)

    # Download data if missing folders
    if cfg.tusz.force_download or (
        not set(cfg.tusz.splits) <= {x.stem for x in raw_edf_folder.iterdir()}
    ):
        logging.info("Downloading data")
        download(
            source=cfg.tusz.source,
            target=cfg.tusz.raw,
            password=os.environ["NEDC_PASSWORD"],
        )

    for split in cfg.tusz.splits:
        logging.info("Creating %s dataset", split.upper())

        root_folder = raw_edf_folder / split
        signals_out_folder = output_folder / split / "signals"

        logging.info("Creating segments dataframe from %s", root_folder)
        segments_df = process_walk(
            root_folder,
            signals_out_folder=signals_out_folder,
            sampling_rate_out=cfg.signals.sampling_rate,
            label_map=OmegaConf.to_container(cfg.labels.map),
            binary=cfg.labels.binary,
        )

        segments_save_path = output_folder / split / "segments.parquet"
        write_parquet(segments_df, segments_save_path)

        logging.info(
            "Created %s dataset - # samples: %d",
            split.upper(),
            len(segments_df.index.droplevel("channel").unique()),
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

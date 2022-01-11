# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.data.tusz.download import download
from src.data.tusz.io import write_parquet
from src.data.tusz.process import process_walk
from src.run import run

################################################################################
# MAIN

# @click.argument("raw-data-folder", type=click.Path(exists=True, path_type=Path))
# @click.argument("processed-data-folder", type=click.Path(path_type=Path))


@run(config_path="config.yaml")
def main(cfg: DictConfig):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    load_dotenv(find_dotenv())

    raw_edf_folder = Path(cfg.data.tusz.raw_edf)
    output_folder = Path(cfg.data.tusz.processed)

    # Download data if missing folders
    if cfg.data.tusz.force_download or (
        not set(cfg.data.tusz.splits) <= {x.stem for x in raw_edf_folder.iterdir()}
    ):
        logging.info("Downloading data")
        download(
            source=cfg.data.tusz.source,
            target=cfg.data.tusz.raw,
            password=os.environ["NEDC_PASSWORD"],
        )

    for split in cfg.data.tusz.splits:
        logging.info("Creating %s dataset", split.upper())

        root_folder = raw_edf_folder / split
        signals_out_folder = output_folder / split / "signals"

        logging.info("Creating clips dataframe from %s", root_folder)
        clips_df = process_walk(
            root_folder,
            signals_out_folder=signals_out_folder,
            sampling_rate_out=cfg.data.signals.sampling_rate,
            diff_channels=cfg.data.signals.diff_channels,
            label_map=OmegaConf.to_container(cfg.data.labels.map),
            binary=cfg.data.labels.binary,
            clip_length=cfg.data.signals.clip_length,
            clip_stride=cfg.data.signals.clip_stride,
        )

        clips_save_path = output_folder / "clips.parquet"
        write_parquet(clips_df, clips_save_path)

        logging.info(
            "Created %s dataset - # samples: %d",
            split.upper(),
            len(clips_df.index.droplevel("channel").unique()),
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

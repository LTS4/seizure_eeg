# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
from pathlib import Path

from omegaconf import OmegaConf

from seiz_eeg.config import DataConf
from seiz_eeg.constants import GLOBAL_CHANNEL
from seiz_eeg.preprocess.io import write_parquet
from seiz_eeg.preprocess.tusz.download import download
from seiz_eeg.preprocess.tusz.process import process_walk
from seiz_eeg.schemas import ClipsLocalDF

################################################################################
# MAIN


def main(cfg: DataConf):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    raw_root_folder = Path(cfg.tusz.raw_edf)
    output_folder = Path(cfg.tusz.processed)

    raw_root_folder.mkdir(parents=True, exist_ok=True)

    # Download data if missing folders
    if cfg.tusz.force_download or (
        not set(cfg.tusz.subsets) <= {x.stem for x in raw_root_folder.iterdir()}
    ):
        logging.info("Downloading data")
        download(
            version=cfg.tusz.version,
            target=cfg.tusz.raw,
            user=cfg.tusz.user,
            password=cfg.tusz.password,
        )

    for split in cfg.tusz.subsets:
        logging.info("Creating %s dataset", split.upper())

        root_folder = raw_root_folder / split
        signals_out_folder = output_folder / split / "signals"

        logging.info("Creating segments dataframe from %s", root_folder)
        segments_df = process_walk(
            root_folder,
            signals_out_folder=signals_out_folder,
            sampling_rate_out=cfg.signals.sampling_rate,
            label_map=OmegaConf.to_container(cfg.labels.map),
            exclude_patients=cfg.tusz.excluded_patients.get(split),
        )

        write_parquet(segments_df, output_folder / split / "segments_local.parquet")
        write_parquet(
            segments_df.xs(key=GLOBAL_CHANNEL, level=ClipsLocalDF.channel),
            output_folder / split / "segments.parquet",
        )

        logging.info(
            "Created %s dataset - # samples: %d",
            split.upper(),
            len(segments_df.index.droplevel("channel").unique()),
        )


if __name__ == "__main__":
    config = OmegaConf.load("data_config.yaml")
    config.merge_with_cli()

    main(config)

# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
import os
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.data.tusz.dataset import make_dataset
from src.data.tusz.download import download
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
        dataset_path = output_folder / split / "data.pt"

        logging.info("Creating %s dataset", split.upper())

        eeg_data = make_dataset(
            root_folder=raw_edf_folder / split,
            output_folder=output_folder / split,
            # Signals options
            clip_length=cfg.data.signals.clip_length,
            clip_stride=cfg.data.signals.clip_stride,
            label_map=OmegaConf.to_container(cfg.data.labels.map),
            binary=cfg.data.labels.binary,
            node_level=cfg.data.labels.node_level,
            # Dataset options
            sampling_rate=cfg.data.signals.sampling_rate,
            window_len=cfg.data.signals.window_len,
            diff_channels=cfg.data.signals.diff_channels,
        )

        logging.info("Created %s dataset - # samples: %d", split.upper(), len(eeg_data))

        torch.save(eeg_data, dataset_path)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

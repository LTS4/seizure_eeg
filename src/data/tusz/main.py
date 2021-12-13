# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.data.tusz.dataset import make_dataset
from src.run import run

logger = logging.getLogger(__name__)


################################################################################
# MAIN

# @click.argument("raw-data-folder", type=click.Path(exists=True, path_type=Path))
# @click.argument("processed-data-folder", type=click.Path(path_type=Path))


@run(config_path="config.yaml")
def main(cfg: DictConfig):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    print(OmegaConf.to_yaml(cfg))

    raw_edf_folder = Path(cfg.data.tusz.raw_edf)
    output_folder = Path(cfg.data.tusz.processed)

    for split in ("dev", "train"):
        logging.info("Creating %s dataset", split.upper())

        eeg_data = make_dataset(
            root_folder=raw_edf_folder / split,
            clip_length=cfg.data.signals.clip_length,
            binary=cfg.data.labels.binary,
            clips_save_path=output_folder / split / "clips.parquet",
            # sampling_rate=cfg.data.signals.sampling_rate,
            # diff_channels=cfg.data.signals.diff_channels,
            # binary=False,
        )

        logging.info("Created %s dataset - # samples: %d", split.upper(), len(eeg_data))

        print(eeg_data[0])


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

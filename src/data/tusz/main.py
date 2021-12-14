# -*- coding: utf-8 -*-
"""Main module for dataset creation"""
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.data.tusz.dataset import make_dataset
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
    raw_edf_folder = Path(cfg.data.tusz.raw_edf)
    output_folder = Path(cfg.data.tusz.processed)

    for split in cfg.data.tusz.splits:
        logging.info("Creating %s dataset", split.upper())

        eeg_data = make_dataset(
            root_folder=raw_edf_folder / split,
            clip_length=cfg.data.signals.clip_length,
            label_map=OmegaConf.to_container(cfg.data.labels.map),
            binary=cfg.data.labels.binary,
            load_existing=cfg.data.load_existing,
            # Dataset options
            clips_save_path=output_folder / split / "clips.parquet",
            sampling_rate=cfg.data.signals.sampling_rate,
            diff_channels=cfg.data.signals.diff_channels,
        )

        logging.info("Created %s dataset - # samples: %d", split.upper(), len(eeg_data))

        for i in range(len(eeg_data)):
            print(i)
            sample = eeg_data[i]
            print("Label:", sample[0])
            print("Signals:", sample[1].shape)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

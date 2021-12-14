"""Pipeline to generate dataset"""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.dataset import EEGDataset
from src.data.tusz.annotations.process import process_annotations
from src.data.tusz.io import write_parquet

################################################################################
# DATASET


def make_dataset(
    root_folder: Path,
    *,
    clip_length: int,
    clip_stride: int,
    label_map: Dict[str, str],
    binary: bool,
    sampling_rate: int,
    node_level: bool,
    diff_channels: bool,
    load_existing: Optional[bool] = False,
    clips_save_path: Optional[Path] = None,
) -> EEGDataset:
    """Create eeg dataset by parsing all files in root_folder"""
    if load_existing and clips_save_path and clips_save_path.exists():
        logging.info("Reading clips dataframe: %s", clips_save_path)
        clips_df = pd.read_parquet(clips_save_path)
    else:
        logging.info("Creating clips dataframe from %s", root_folder)
        clips_df = process_annotations(
            root_folder,
            label_map=label_map,
            binary=binary,
            clip_length=clip_length,
            clip_stride=clip_stride,
        )

        if clips_save_path:
            write_parquet(clips_df, clips_save_path)

    return EEGDataset(
        clips_df,
        sampling_rate=sampling_rate,
        node_level=node_level,
        diff_channels=diff_channels,
    )

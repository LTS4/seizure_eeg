"""Pipeline to generate dataset"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.tusz.constants import (
    FILE_METADATA,
    FILE_SEGMENTS_DF,
    FILE_SIGNAL_DIFF,
    FILE_SIGNAL_REF,
    GLOBAL_CHANNEL,
)
from src.data.tusz.io import list_all_edf_files, write_parquet
from src.data.tusz.labels.io import read_labels
from src.data.tusz.labels.process import get_channels, time_to_samples
from src.data.tusz.signals.io import read_eeg_signals
from src.data.tusz.signals.process import get_diff_signals, get_resampled_signals

logger = logging.getLogger(__name__)


def process_dataset(
    root_folder: Path,
    output_folder: Path,
    *,
    sampling_rate: int,
    binary: bool,
    diff_channels: bool,
    force_rewrite: Optional[bool] = False,
):
    """Precess every file in the root_folder tree"""
    file_list = list_all_edf_files(root_folder)

    nb_errors_skipped = 0
    nb_existing = 0

    # Write metadata
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder / FILE_METADATA, "w") as f:
        json.dump(
            dict(
                sampling_rate=sampling_rate,
                file_signal_ref=FILE_SIGNAL_DIFF,
                file_signal_diff=FILE_SIGNAL_DIFF,
            ),
            f,
        )

    annotations_list = []

    for edf_path in tqdm(file_list, desc="Processing dataset"):
        try:
            session_id = edf_path.stem
            session_folder = output_folder / session_id
            os.makedirs(session_folder, exist_ok=True)

            signals = get_resampled_signals(*read_eeg_signals(edf_path), sampling_rate)

            # fmt: off
            annotations = (
                read_labels(edf_path, binary)
                .pipe(time_to_samples, len(signals))
            )
            # fmt: on

            annotations_list.append(annotations)

            label_channels = get_channels(annotations).drop(GLOBAL_CHANNEL)

            # Save signals to files
            if diff_channels:
                signal_diff_path = session_folder / FILE_SIGNAL_DIFF
                try:
                    diff_signals = get_diff_signals(signals, label_channels)
                except KeyError as err:
                    raise KeyError(
                        f"KeyError in session '{session_id}' with signal columns: {signals.columns.tolist()}"
                    ) from err

                if not write_parquet(diff_signals, signal_diff_path, force_rewrite=force_rewrite):
                    nb_existing += 1
            else:
                signal_ref_path = session_folder / FILE_SIGNAL_REF
                if not write_parquet(signals, signal_ref_path, force_rewrite=force_rewrite):
                    nb_existing += 1

        except (IOError, AssertionError) as err:
            logger.info("Skipping file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err)
            nb_errors_skipped += 1

    # Create segments database and save
    pd.concat(annotations_list, ignore_index=False).to_parquet(
        output_folder / FILE_SEGMENTS_DF,
    )

    if nb_errors_skipped:
        logger.warning("Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped)
    if nb_existing:
        logger.warning("Skipped %d files which existed already, set level to INFO for details", nb_existing)

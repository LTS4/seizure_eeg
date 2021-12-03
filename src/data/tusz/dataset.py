"""Pipeline to generate dataset"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.tusz.constants import FILE_SEGMENTS_DF, FILE_SIGNALS, GLOBAL_CHANNEL
from src.data.tusz.io import list_all_edf_files, write_parquet
from src.data.tusz.labels.io import read_labels
from src.data.tusz.labels.process import get_channels, time_to_samples
from src.data.tusz.signals.io import read_eeg_signals
from src.data.tusz.signals.process import process_signals

logger = logging.getLogger(__name__)


def process_session(edf_path: Path, *, sampling_rate: int, binary: bool, diff_channels: bool):
    """Retrieve annotations and samples for one session."""

    annotations = read_labels(edf_path, binary)

    if diff_channels:
        diff_labels = get_channels(annotations).drop(GLOBAL_CHANNEL)
    else:
        diff_labels = None

    signals = process_signals(
        *read_eeg_signals(edf_path), sampling_rate_out=sampling_rate, diff_labels=diff_labels
    )

    annotations = annotations.pipe(time_to_samples, len(signals))

    return annotations, signals


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

    annotations_list = []

    for edf_path in tqdm(file_list, desc="Processing dataset"):
        try:
            annotations, signals = process_session(
                edf_path, sampling_rate=sampling_rate, binary=binary, diff_channels=diff_channels
            )

            annotations_list.append(annotations)

            if not write_parquet(
                signals,
                path=output_folder / edf_path.stem / FILE_SIGNALS,
                force_rewrite=force_rewrite,
            ):
                nb_existing += 1

        except (IOError, AssertionError) as err:
            logger.info(
                "Skipping file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err
            )
            nb_errors_skipped += 1

    # Create segments database and save
    pd.concat(annotations_list, ignore_index=False).to_parquet(
        output_folder / FILE_SEGMENTS_DF,
    )

    if nb_errors_skipped:
        logger.warning(
            "Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped
        )
    if nb_existing:
        logger.warning(
            "Skipped %d files which existed already, set level to INFO for details", nb_existing
        )

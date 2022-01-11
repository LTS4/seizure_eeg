"""Pipeline to generate dataset"""
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm

from src.data.schemas import ClipsDF
from src.data.tusz.annotations.process import make_clips, process_annotations
from src.data.tusz.io import list_all_edf_files
from src.data.tusz.signals.io import read_eeg_signals
from src.data.tusz.signals.process import process_signals

################################################################################
# DATASET


def process_walk(
    root_folder: Path,
    *,
    signals_out_folder: Path,
    sampling_rate_out: int,
    diff_channels: bool,
    label_map: Dict[str, int],
    binary: bool,
    clip_length: int,
    clip_stride: int,
) -> DataFrame[ClipsDF]:
    """Precess every file in the root_folder tree"""
    logger = logging.getLogger(__name__)

    if not signals_out_folder.exists():
        signals_out_folder.mkdir(parents=True)
    elif not signals_out_folder.is_dir():
        raise ValueError(f"Target exists, but is not a directory ({signals_out_folder})")

    nb_errors_skipped = 0

    annotations_list = []

    for edf_path in tqdm(list_all_edf_files(root_folder), desc=f"{root_folder}"):
        try:
            # Convert signals
            signals_path = (signals_out_folder / edf_path.stem).with_suffix(".parquet")

            signals = process_signals(
                *read_eeg_signals(edf_path),
                sampling_rate_out=sampling_rate_out,
                diff_channels=diff_channels,
            )

            signals.to_parquet(signals_path)

            # Process annotations
            annotations_list.append(
                process_annotations(
                    edf_path,
                    label_map=label_map,
                    binary=binary,
                    signals_path=signals_path,
                    sampling_rate=sampling_rate_out,
                )
            )

        except (IOError, AssertionError) as err:
            logger.info(
                "Excluding file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err
            )
            nb_errors_skipped += 1

    if nb_errors_skipped:
        logger.warning(
            "Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped
        )

    # Make clips from annotations. Faster since in batch
    return pd.concat(annotations_list, ignore_index=False).pipe(
        make_clips, clip_length=clip_length, clip_stride=clip_stride
    )

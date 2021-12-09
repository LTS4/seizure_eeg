"""Pipeline to generate dataset"""
import logging
from pathlib import Path

import pandas as pd
from pandera.typing import DataFrame
from tqdm import tqdm

from src.data.dataset import EEGDataset
from src.data.schemas import AnnotationDF
from src.data.tusz.constants import FILE_SEGMENTS_DF
from src.data.tusz.io import list_all_edf_files, write_parquet
from src.data.tusz.labels.io import read_labels

logger = logging.getLogger(__name__)


def process_annotations(
    root_folder: Path,
    *,
    binary: bool,
) -> DataFrame[AnnotationDF]:
    """Precess every file in the root_folder tree"""
    file_list = list_all_edf_files(root_folder)

    nb_errors_skipped = 0

    annotations_list = []

    for edf_path in tqdm(file_list, desc=f"{root_folder}"):
        try:

            annotations = read_labels(edf_path, binary)
            annotations_list.append(annotations)

        except (IOError, AssertionError) as err:
            logger.info(
                "Skipping file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err
            )
            nb_errors_skipped += 1

    if nb_errors_skipped:
        logger.warning(
            "Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped
        )

    return pd.concat(annotations_list, ignore_index=False)


def make_clips(
    annotations: DataFrame[AnnotationDF],
    clip_length: int,
) -> DataFrame[AnnotationDF]:
    "Split annotations dataframe in dataframe of non-overlapping clips"
    annotations = annotations.reset_index()
    start_times, end_times = annotations["start_time"], annotations["end_time"]

    out_list = []
    for clip_idx in range(int(end_times.max() / clip_length)):
        clip_start = clip_idx * clip_length
        clip_end = (clip_idx + 1) * clip_length

        filter = (start_times <= clip_start) & (clip_end <= end_times)

        copy_vals = annotations[filter].copy()
        copy_vals[["segment", "start_time", "end_time"]] = clip_idx, clip_start, clip_end
        out_list.append(copy_vals)

    return pd.concat(out_list).set_index(["channel", "patient", "session", "segment"]).sort_index()


################################################################################
# DATASET


def make_dataset(
    root_folder: Path,
    *,
    binary: bool,
) -> EEGDataset:
    clips = make_clips(annotations=process_annotations(root_folder, binary=binary))

    # # Create segments database and save
    # write_parquet(
    #     output_folder / FILE_SEGMENTS_DF,
    # )

    pass

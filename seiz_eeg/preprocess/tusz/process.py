"""Pipeline to generate dataset"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from pandera.typing import DataFrame
from tqdm import tqdm

from seiz_eeg.preprocess.io import list_all_edf_files
from seiz_eeg.preprocess.tusz.annotations.process import process_annotations
from seiz_eeg.preprocess.tusz.signals.io import read_eeg_signals
from seiz_eeg.preprocess.tusz.signals.process import preprocess_signals
from seiz_eeg.schemas import ClipsLocalDF

################################################################################
# DATASET


def params_changed(params_path: Path, **kwargs) -> bool:
    """Check wheter parameters in *params_path* are equal to *kwargs*.
    If not, overwrite *params_path* with new params.
    """
    if params_path.exists():
        with params_path.open("r", encoding="utf-8") as file:
            old_params = yaml.safe_load(file)

        if kwargs == old_params:
            return False

    with params_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(kwargs, file)
    return True


def process_walk(
    root_folder: Path,
    *,
    signals_out_folder: Path,
    sampling_rate_out: int,
    label_map: Dict[str, int],
    binary: bool,
    exclude_patients: Optional[List[str]] = None,
) -> DataFrame[ClipsLocalDF]:
    """Precess every file in the root_folder tree and return the dataset of EEG segments"""
    logger = logging.getLogger(__name__)

    if not signals_out_folder.exists():
        signals_out_folder.mkdir(parents=True)
    elif not signals_out_folder.is_dir():
        raise ValueError(f"Target exists, but is not a directory ({signals_out_folder})")

    nb_errors_skipped = 0

    annotations_list: List[ClipsLocalDF] = []

    reprocess = params_changed(
        signals_out_folder / "signals_params.yaml",
        sampling_rate_out=sampling_rate_out,
    )

    for edf_path in tqdm(list_all_edf_files(root_folder), desc=f"{root_folder}"):
        try:
            signals_path: Path = (signals_out_folder / edf_path.stem).with_suffix(".parquet")

            if exclude_patients is None or edf_path.parents[1].stem not in exclude_patients:

                if not signals_path.exists() or reprocess:
                    # Process signals and save them
                    preprocess_signals(
                        *read_eeg_signals(edf_path),
                        sampling_rate_out=sampling_rate_out,
                    ).to_parquet(signals_path)

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

        except (IOError, AssertionError, ValueError) as err:
            logger.info(
                "Excluding file %s wich raises %s: \n\t%s", edf_path, type(err).__name__, err
            )
            nb_errors_skipped += 1

    if nb_errors_skipped:
        logger.warning(
            "Skipped %d files raising errors, set level to INFO for details", nb_errors_skipped
        )

    return pd.concat(annotations_list, ignore_index=False)

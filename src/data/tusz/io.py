"""I/O utilities"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandera.typing import DataFrame

from src.data.tusz.utils import lower_na

logger = logging.getLogger(__name__)


def read_seiz_vocabulary(file_path: Path) -> DataFrame:
    return (
        pd.read_excel(file_path)[["Class Code", "Class No."]]
        .rename({"Class Code": "label_str", "Class No.": "label_int"}, axis="columns")
        .pipe(lower_na, "label_str")
    )


def list_all_edf_files(root_path: Path) -> List[Path]:
    """Walk down from *root_path* and retieve a list of all files with ``.edf`` extension.

    Args:
        root_path (Path): Path to root folder of edf dataset

    Returns:
        List[Path]: List of all relative paths to ``.edf`` files.
    """
    filelist = [
        Path(dirpath) / file
        for dirpath, _, filenames in os.walk(root_path, followlinks=True)
        for file in filenames
        if file.endswith("edf")
    ]

    return filelist


def write_parquet(df: DataFrame, path: Path, force_rewrite: Optional[bool] = True) -> bool:
    """Write dataframe to parquet and return ``True`` if succeded.
    If not *force_rewrite* log if file exists and return ``False``"""

    if not force_rewrite and path.exists():
        logger.info("Skipping existing file: %s", path)
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    return True

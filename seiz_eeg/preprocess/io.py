"""I/O utilities"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandera.typing import DataFrame

logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=50)
def read_parquet(file_path):
    return pd.read_parquet(file_path)

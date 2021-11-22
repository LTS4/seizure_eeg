"""Utility functions for data retrieval"""
import os
from pathlib import Path
from typing import List


def list_all_edf_files(root_path: Path) -> List[Path]:
    """Walk down from *root_path* and retieve a list of all files with ``.edf`` extension.

    Args:
        root_path (Path): Path to root folder of edf dataset

    Returns:
        List[Path]: List of all relative paths to ``.edf`` files.
    """
    filelist = [
        Path(dirpath) / file
        for dirpath, _, filenames in os.walk(root_path)
        for file in filenames
        if file.endswith("edf")
    ]

    return filelist

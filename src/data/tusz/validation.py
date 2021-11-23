"""Utilities to verify consistency in data"""
from pathlib import Path
from typing import List


def verify_filenames(edf_list: List[Path], annotations_list: List[str]) -> bool:
    """Check whether the *edf_list*  obtained by walking the folder structure corresponds to the
    list of files provided in the annotation xls file (*annotation_list*)

    Args:
        edf_list (List[Path]): List of ``.edf`` files in the data folder.
        annotations_list (List[str]): List of files in the columns of the annotation file.

    Returns:
        bool: True if the lists are equal
    """
    raise NotImplementedError()

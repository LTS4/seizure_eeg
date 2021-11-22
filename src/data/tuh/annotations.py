"""Dataclasses related to annotations and reader function"""

from collections import defaultdict
import re
from ast import literal_eval
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

@dataclass
class Labels:
    """Dataclass containing seizure label and start/stop times"""
    start: float
    stop: float
    label: str

@dataclass
class Annotations:
    """
    Dataclass containing:
        - user_id
        - date
        - session_id
        - file_name
        - dictionary of *montage* -> `Labels`
    """
    user_id: str
    date: datetime
    session_id: str
    file_id: str

    labels_map: Dict[str, List[Labels]]


def extract_session_date(string: str) -> Tuple[str, datetime]:
    """From a string of format ``s\\d{3}_YYYY_MM_DD`` extract date and session id."""
    match = re.search(
        r"s(\d{3})_(\d{4}_\d{2}_\d{2})",
        string,
    )
    session_id = match.group(1)

    date = datetime.strptime(match.group(2), "%Y_%m_%d")

    return session_id, date


def read_tse(tse_path: Path) -> List[Labels]:
    """Extract global labels and timestamps from .tse file"""
    labels = []
    for line in tse_path.read_text().splitlines():
        if line and not line.startswith("version"):
            split = line.split(" ")
            labels.append(
                Labels(split[0], split[1], split[2])
            )

    return labels


REGEX_LABEL = (
    r"\{(?P<level>\d+), (?P<unk>\d+), (?P<start>\d+\.\d{4}), "
    r"(?P<stop>\d+\.\d{4}), (?P<montage_n>\d+), (?P<oh_sym>\[.*\])\}"
)
REGEX_SYMBOLS = r"symbols\[(?P<level>\d+)\].*(?P<sym_dict>\{.*\})"
REGEX_MONTAGE = r"(?P<num>\d+), (?P<montage>\w+\d*-\w+\d*):"

def read_lbl(lbl_path: Path) -> Dict[str, List[Labels]]:
    """Parse `.lbl[_bi]` file.

    Args:
        lbl_path (Path): Path to `.lbl[_bi]` file.

    Returns:
        Dict[str, List[Labels]]: Dictionary with montages (i.e. strings with "<electrode1>-<electrode2>") as keys.
            Itemes are lists of `Labels`, i.e. dataclasses containing start/stops times and seizure labels.
    """
    montages = []
    symbols = []
    labels = defaultdict(list)

    for line in lbl_path.read_text().splitlines():
        # All montage lines come first
        if line.startswith("montage"):
            match = re.search(REGEX_MONTAGE, line)
            num = int(match.group("num"))
            assert int(num) == len(montages), f"Missing montage {len(montages)} in file {lbl_path}"

            montages.append(match.group("montage"))

        # Then come symbols. Multiple levels of annotations are possible.
        if line.startswith("symbols"):
            match = re.search(REGEX_SYMBOLS, line)

            level = int(match.group("level"))

            assert level == len(symbols), f"Missing symbol level {len(symbols)} in file {lbl_path}"
            symbols.append(literal_eval(match.group("sym_dict")))

        # Finally come labels. At this point montages and symbols should be defined.
        if line.startswith("label"):
            match = re.search(REGEX_LABEL, line)

            level, montage_n = map(int, match.group("level", "montage_n"))
            sym_int = np.nonzero(literal_eval(match.group("oh_sym")))[0].item()
            start, stop = map(float, match.group("start", "stop"))

            labels[montages[montage_n]].append(
            Labels(start, stop, label=symbols[level][sym_int])
            )

    # We convert labels to a astandard dict to prevent silent failure on missing montages
    return dict(labels)


def read_labels(edf_path: Path, binary: bool) -> Dict[str, List[Labels]]:
    """Retrieve seizure labels parsing the ``.tse[_bi]`` and the ``.lbl[_bi]`` files corresponding to the ``.edf``
    file at *file_path*.

    Args:
        edf_path (Path): Path to the ``.edf`` file
        binary (bool): Wheter to  use the ``.*_bi`` version of the label files,
            which differentiate only *bkgd* vs *seiz*

    Returns:
        Dict[str, List[Labels]]: dictionary of montage symbols and list of corresponding `Labels`
    """
    tse_suffix = ".tse_bi" if binary else ".tse"
    lbl_suffix = ".lbl_bi" if binary else ".lbl"

    tse_path = edf_path.with_suffix(tse_suffix)
    assert tse_path.exists(), f"File not found: {tse_path}"

    lbl_path = edf_path.with_suffix(lbl_suffix)
    assert lbl_path.exists(), f"File not found: {lbl_path}"

    *_, labels = read_lbl(lbl_path)
    labels["general"] = read_tse(tse_path)

    return labels



def get_edf_annotations(edf_path: Path, binary: bool) -> Annotations:
    """Use edf path to retrieve annotations.
    Gathered info:
        - user id
        - session id
        - date
        - time-stamps to label mapping:
            - general
            - per node

    Args:
        edf_path (Path): [description]
        binary (bool): [description]

    Raises:
        NotImplementedError: [description]

    Returns:
        Dict[str, Any]: [description]
    """
    file_id = edf_path.stem
    user_id = edf_path.parents[1].stem
    session_id, date = extract_session_date(edf_path.parents[0].stem)
    labels = read_labels(edf_path, binary)

    return Annotations(
        user_id, date, session_id, file_id=file_id,
        labels_map=labels,
    )

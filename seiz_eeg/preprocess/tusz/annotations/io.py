"""I/O functions for annotations files (lbl and tse)"""

import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from pandera import check_types
from pandera.typing import DataFrame

from seiz_eeg.constants import GLOBAL_CHANNEL
from seiz_eeg.preprocess.tusz.constants import REGEX_LABEL, REGEX_MONTAGE, REGEX_SYMBOLS
from seiz_eeg.preprocess.tusz.utils import check_label, concat_labels
from seiz_eeg.schemas import ClipsDF, LabelDF


@check_types
def read_tse(tse_path: Path) -> DataFrame[LabelDF]:
    """Extract global labels and timestamps from .tse file"""
    labels = []

    for line in tse_path.read_text().splitlines():
        if line and not line.startswith("version"):
            split = line.split(" ")
            labels.append(
                dict(
                    channel=GLOBAL_CHANNEL,
                    label=check_label(split[2]),
                    start_time=float(split[0]),
                    end_time=float(split[1]),
                )
            )

    return concat_labels(labels)


@check_types
def read_lbl(lbl_path: Path) -> DataFrame[LabelDF]:
    """Parse `.lbl[_bi]` file.

    Args:
        lbl_path (Path): Path to `.lbl[_bi]` file.

    Returns:
        Dict[str, List[Labels]]: Dictionary with montages (i.e. strings with
            ``"<electrode1>-<electrode2>"`` ) as keys. Items are lists of `Labels`, i.e. dataclasses
            containing start/ends times and seizure labels.
    """
    montages = []
    symbols = []
    labels = []

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
            start, end = map(float, match.group("start", "end"))

            labels.append(
                dict(
                    channel=montages[montage_n],
                    label=check_label(symbols[level][sym_int]),
                    start_time=start,
                    end_time=end,
                )
            )

    return concat_labels(labels)


@check_types
def read_csv(csv_path: Path) -> DataFrame[LabelDF]:
    if not csv_path.exists():
        raise IOError(f"File not found: {csv_path}")

    return concat_labels(
        pd.read_csv(csv_path, header=0, comment="#", index_col=False)
        .drop(columns="confidence")
        .rename(columns={"stop_time": "end_time"})
    )


@check_types
def read_labels(edf_path: Path) -> DataFrame[LabelDF]:
    """Retrieve seizure labels parsing the ``.tse[_bi]`` and the ``.lbl[_bi]`` files corresponding
    to the ``.edf`` file at *file_path*.

    Args:
        edf_path (Path): Path to the ``.edf`` file
        binary (bool): Wheter to  use the ``.*_bi`` version of the label files,
            which differentiate only *bkgd* vs *seiz*

    Raises:
        IOError: If one of the labels files is missing

    Returns:
        DataFrame[LabelSchema]: Dataframe of montage symbols with labels and timestamps
    """
    return pd.concat(
        [
            read_csv(edf_path.with_suffix(".csv")),
            read_csv(edf_path.with_suffix(".csv_bi")).replace("TERM", "global"),
        ]
    )


####################################################################################################
# Reference files


def read_ref(doc_path: Path) -> DataFrame:
    """Read the ``ref_*.txt`` files in the doc folder"""
    df_list = []
    for file in doc_path.glob("ref_*.txt"):
        df = pd.read_csv(
            file,
            sep=" ",
            names=[
                ClipsDF.session,
                ClipsDF.start_time,
                ClipsDF.end_time,
                ClipsDF.label,
                "prob",
            ],
        )

        df["split"] = file.stem.replace("ref_", "")
        df_list.append(df)

    assert len(df_list) > 0, "No ref file found"

    return pd.concat(df_list, ignore_index=True)


def path_to_id_single(x):
    path = Path(x)
    return pd.Series((path.parents[1].stem, path.stem))


def path_to_id(df):
    df[[ClipsDF.patient, ClipsDF.session]] = df[ClipsDF.session].apply(path_to_id_single)
    return df


def rename_columns(df, columns):
    df.columns = columns
    return df


def parse_calibration(df_slice):
    """Parse calibration"""
    df = (
        df_slice.iloc[3:]
        .pipe(
            rename_columns,
            columns=[ClipsDF.session, ClipsDF.start_time, ClipsDF.end_time],
        )
        .dropna()
        .reset_index(drop=True)
        .pipe(path_to_id)
    )
    df["split"] = df_slice.iloc[0, 0]
    return df


def read_calibration(doc_path: Path) -> DataFrame:
    """Read calibration sheet from documentation excel file"""
    calibration = pd.read_excel(
        doc_path / "seizures_v36r.xlsx", sheet_name="Calibration", header=None
    )

    calibration = pd.concat(
        [parse_calibration(calibration.iloc[:, 4 * i : 4 * i + 3]) for i in range(3)],
        ignore_index=True,
    )

    return calibration

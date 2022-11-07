"""I/O utilities"""
from pathlib import Path

import pandas as pd
from pandera.typing import DataFrame

from seiz_eeg.preprocess.tusz.utils import lower_na


def read_seiz_vocabulary(file_path: Path) -> DataFrame:
    return (
        pd.read_excel(file_path)[["Class Code", "Class No."]]
        .rename({"Class Code": "label_str", "Class No.": "label_int"}, axis="columns")
        .pipe(lower_na, "label_str")
    )

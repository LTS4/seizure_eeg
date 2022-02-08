"""Utility functions for data retrieval"""
import re
from datetime import datetime
from typing import List

import pandas as pd
from pandera.typing import DataFrame

from src.data.schemas import LabelDF
from src.data.tusz.constants import TYPOS


def lower_na(df, column: str):
    df[column] = df[column].fillna("null").apply(str.lower)
    return df


def concat_labels(labels_list: List[dict]) -> DataFrame[LabelDF]:
    """Create a dataset from a list of labels dictionaries"""
    # pd.DataFrame(columns=["channel", "label", "start_time", "end_time"])
    df = pd.DataFrame(labels_list)
    return df.assign(segment=df.groupby("channel").cumcount())


def check_label(label: str) -> str:
    return TYPOS.get(label, label)


def extract_session_date(string: str) -> datetime:
    """From a string of format ``s\\d{3}_YYYY_MM_DD`` extract date and session id."""
    match = re.search(
        r"s\d{3}_(\d{4}_\d{2}_\d{2})",
        string,
    )
    date = datetime.strptime(match.group(1), "%Y_%m_%d")

    return date

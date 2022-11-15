"""Preprocessing pipleine for CHB-MIT"""
from pathlib import Path

import pandas as pd

from seiz_eeg.preprocess.chbmit.annotations import parse_patient
from seiz_eeg.preprocess.chbmit.signals import convert_all_sessions
from seiz_eeg.preprocess.io import write_parquet


def main(cfg):
    """Preprocessing pipeline for CHB-MIT

    Args:
        cfg (_type_): Configuration
    """
    raw_root = Path(cfg.raw_root)

    pat_info = pd.read_csv(
        raw_root / "SUBJECT-INFO",
        sep="\t",
        index_col="Case",
    )
    patients = pat_info.index.unique()

    df = pd.concat([parse_patient(raw_root, patient) for patient in patients]).sort_index()

    if cfg.get("convert", False):
        df = convert_all_sessions(df, Path(cfg.processed_root) / "signals")

    write_parquet(df, Path(cfg.processed_root) / "segments.parquet")

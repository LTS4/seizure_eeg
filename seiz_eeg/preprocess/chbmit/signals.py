"""Signals conversion utility"""

from pathlib import Path

import numpy as np
from pandas import IndexSlice as idx
from pandera.typing import DataFrame
from tqdm import tqdm

from seiz_eeg.preprocess.io import ParallelEdfReader, write_parquet
from seiz_eeg.schemas import ClipsDF

from .annotations import replace_all
from .constants import REPLACEMENTS


def convert_all_sessions(df: DataFrame[ClipsDF], out_signals_root: Path) -> DataFrame[ClipsDF]:
    """Convert signals of all sesisons in :arg:`df` from EDF to parquet files.

    Args:
        df (DataFrame[ClipsDF]): Segments dataframe linking to edf files
        out_signals_root (Path): Path to directory where to save the converted files

    Returns:
        DataFrame[ClipsDF]: Segments dataframe linking to parquet files
    """
    df_copy = df.copy()
    for (pat, sess), group in tqdm(
        df.groupby(level=[ClipsDF.patient, ClipsDF.session]),
        desc="Converting signals",
    ):
        edf_path = group[ClipsDF.signals_path].unique().item()
        channels = group["channels"][0]
        assert np.all([np.all(chls == channels) for chls in group["channels"]])

        reader = ParallelEdfReader(
            edf_path,
            format_channel_name=lambda x: replace_all(x, REPLACEMENTS),
            channels_to_read=channels,
        )

        out_path = out_signals_root / pat / f"s{sess}.parquet"
        write_parquet(reader.read_signals(), out_path)

        df_copy.loc[idx[pat, sess, :], ClipsDF.signals_path] = str(out_path)

    return df_copy

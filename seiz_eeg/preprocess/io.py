"""I/O utilities"""
import logging
import os
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from pandera.typing import DataFrame
from pyedflib import EdfReader

from seiz_eeg.schemas import SignalsDF

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


class ParallelEdfReader:
    """EDF+ reader with parallel processing.

    Args:
        edf_path (str): Path to edf file
    """

    def __init__(
        self,
        edf_path: str,
        format_channel_name: Optional[Callable[[str], str]] = None,
        channels_to_read: Optional[List[str]] = None,
    ) -> None:
        self.edf_path = edf_path

        edf_reader = EdfReader(str(self.edf_path))
        self.channel_map = {}
        for i, channel in enumerate(edf_reader.getSignalLabels()):
            channel = format_channel_name(channel)
            if channels_to_read is None or channel in channels_to_read:
                self.channel_map[channel] = i

        # self.n_channels = edf_reader.signals_in_file
        # self.nb_samples = edf_reader.getNSamples()

        self.sampling_rates = edf_reader.getSampleFrequencies()

    def read_channel(self, index: int):
        edf_reader = EdfReader(str(self.edf_path))
        return edf_reader.readSignal(index)

    def read_signals(self) -> DataFrame[SignalsDF]:
        """Read signals from EDF and return Dataframe"""
        # def read_signals(self) -> List[NDArray[np.float_]]:

        with Pool(10) as pool:
            signals = pool.map(self.read_channel, self.channel_map.values())

        return pd.DataFrame(data=dict(zip(self.channel_map.keys(), signals)))

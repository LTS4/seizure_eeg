"""Pandera schemas for data validation"""
import pandera as pa
from pandera.typing import DateTime, Index, Series

from src.data.tusz.constants import CHANNELS, REGEX_SIGNAL_CHANNELS, SIGNAL_CHANNELS_FMT


class LabelDF(pa.SchemaModel):
    """
    ======= ======= ===== ========== ========
    Columns
    ------------------------------------------
    segment channel label start_time end_time
    ======= ======= ===== ========== ========
    """

    segment: Series[int]
    channel: Series[str]

    label: Series[str]
    start_time: Series[float]
    end_time: Series[float]


class AnnotationDF(pa.SchemaModel):
    """Dataframe for EEG annotations:

    ======= ======= ======= =======  ===== ========== ======== ==== ========
    Multiindex                       Columns
    -------------------------------  ---------------------------------------
    patient session segment channel  label start_time end_time date edf_path
    ======= ======= ======= =======  ===== ========== ======== ==== ========
    """

    patient: Index[str]
    session: Index[str]
    segment: Index[int]
    channel: Index[str]

    label: Series[int]
    start_time: Series[float]
    end_time: Series[float]
    date: Series[DateTime]
    edf_path: Series[str]
    # sampling_rate: Series[int]

    class Config:
        # pylint: disable-all
        name = "EEGSchema"
        ordered = True

        multiindex_name = "segment_id"
        multiindex_strict = True
        multiindex_coerce = True


class SignalsDF(pa.SchemaModel):
    """Class for eeg signals: columns are channels names and rows are samples"""

    channel: Series[float] = pa.Field(alias="|".join(CHANNELS), regex=True)

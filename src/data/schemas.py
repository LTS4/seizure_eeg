"""Pandera schemas for data validation"""
import pandera as pa
from pandera.typing import DateTime, Index, Series

from src.data.tusz.constants import REGEX_SIGNAL_CHANNELS


class LabelDF(pa.SchemaModel):
    """
    ======= =======  ===== ========== ========
    Columns
    ------------------------------------------
    channel segment  label start_time end_time
    ======= =======  ===== ========== ========
    """

    channel: Series[str]
    segment: Series[int]

    label: Series[str]
    start_time: Series[float]
    end_time: Series[float]


class AnnotationDF(pa.SchemaModel):
    """Dataframe for EEG annotations:

    ======= ======= ======= =======  ===== ========== ======== =============
    Multiindex                       Columns
    -------------------------------  ---------------------------------------
    patient session channel segment  label start_time end_time date edf_path
    ======= ======= ======= =======  ===== ========== ======== ==== ========
    """

    channel: Index[str]
    patient: Index[str]
    session: Index[str]
    segment: Index[int]

    label: Series[str]
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

    channel: Series[float] = pa.Field(alias=REGEX_SIGNAL_CHANNELS, regex=True)

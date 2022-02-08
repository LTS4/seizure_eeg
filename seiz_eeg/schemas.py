"""Pandera schemas for data validation"""
import pandera as pa
from pandera.typing import DateTime, Index, Series

from seiz_eeg.tusz.constants import CHANNELS, MONTAGES


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


class ClipsDF(pa.SchemaModel):
    """Dataframe for EEG annotations:

    ======= ======= ======= =======  ===== ========== ======== ==== ============= ============
    Multiindex                       Columns
    -------------------------------  ---------------------------------------------------------
    patient session segment channel  label start_time end_time date sampling_rate signals_path
    ======= ======= ======= =======  ===== ========== ======== ==== ============= ============
    """

    patient: Index[str]
    session: Index[str]
    segment: Index[int]
    channel: Index[str]

    label: Series[int]
    start_time: Series[float]
    end_time: Series[float]
    date: Series[DateTime]

    sampling_rate: Series[int]
    signals_path: Series[str]

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


class SignalsDiffDF(pa.SchemaModel):
    """Class for eeg signals: columns are names of channels diffs and rows are samples"""

    channel: Series[float] = pa.Field(alias="|".join(MONTAGES), regex=True)


class DigitalSignalsMeta(pa.SchemaModel):
    """Class for digital signals metadata"""

    channel: Index[str]

    ph_max: Series[float]
    ph_min: Series[float]
    d_max: Series[float]
    d_min: Series[float]

"""Pandera schemas for data validation"""
import pandera as pa
from pandera.typing import Index, Series, DateTime


class LabelSchema(pa.SchemaModel):
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


class AnnotationSchema(pa.SchemaModel):
    """Dataframe for EEG annotations:

    ======= ======= ======= =======  ===== ========== ======== ====
    Multiindex                       Columns
    -------------------------------  ------------------------------
    patient session channel segment  label start_time end_time date
    ======= ======= ======= =======  ===== ========== ======== ====
    """

    patient: Index[str]
    session: Index[str]
    channel: Index[str]
    segment: Index[int]

    label: Series[str]
    start_time: Series[float]
    end_time: Series[float]
    date: Series[DateTime]
    # file_path: Series[object]
    # sampling_rate: Series[int]

    class Config:
        # pylint: disable-all
        name = "EEGSchema"
        ordered = True

        multiindex_name = "segment_id"
        multiindex_strict = True
        multiindex_coerce = True

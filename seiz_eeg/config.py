"""Dataclasses illustrating the content of the config file"""
# pylint: disable=missing-class-docstring
from dataclasses import dataclass
from typing import Dict, List, Optional

################################################################################
# DATA CONFIG ##################################################################


@dataclass
class DataSourceConf:
    version: str

    force_download: bool

    raw: str
    raw_edf: str
    raw_doc: str

    processed: str

    splits: List[str]

    excluded_patients_dev: List[str]
    excluded_patients_train: List[str]


@dataclass
class DataLabelsConf:
    map: Dict[str, int]
    binary: bool


@dataclass
class DataSignalsConf:
    diff_channels: bool
    sampling_rate: int  # Hz
    clip_length: float  # seconds
    clip_stride: float  # seconds
    window_len: float  # seconds

    # Interval [min, max] - include all with [null] or switch off with null
    fft_coeffs: Optional[List[Optional[int]]]

    node_level: bool


@dataclass
class DataConf:
    raw_path: str
    processed_path: str

    tusz: DataSourceConf
    labels: DataLabelsConf

    signals: DataSignalsConf

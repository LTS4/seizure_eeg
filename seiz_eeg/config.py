"""Dataclasses illustrating the content of the config file"""
# pylint: disable=missing-class-docstring
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

################################################################################
# DATA CONFIG ##################################################################


@dataclass
class DataSourceConf:
    """Dataclass for parameters specific to one Dataset

    Attributes:
        version (str): Dataset version
        force_download (bool): Download data even if they are already present
        raw (str): Path where to save raw data
        processed (str): Path where to save preprocessed data
        subsets (List[str]): List of subsets to include in preprocessing
            (e.g. ``["train", "test"]``)
        excluded_patients (Dict[str, List[str]]): Map from subset to list of
            patients to exclude from it.
    """

    version: str

    force_download: bool

    raw: str
    raw_edf: str
    raw_doc: str

    processed: str

    subsets: List[str]

    excluded_patients: Dict[str, List[str]]


@dataclass
class DataLabelsConf:
    """Dataclass of seizure labels specifications

    Attributes:
        map (Dict[str, int]): Map from string seizure codes to integers,
            e.g. ``bkgd -> 0`` and ``fnsz -> 1``
        binary (bool): Wheter to read binary labels
    """

    map: Dict[str, int]
    binary: bool


@dataclass
class DataSignalsConf:
    """Dataclass of options for signals and clips processing

    Attributes:
        diff_channels (bool): Wheter to compute channels diffrerences,
            e.g. "T3-T5", "P4-O2", etc.
        sampling_rate (int): Desired sampling rate, in Hz
        clip_length (float): Lenght of clips to extract, in seconds
        clip_stride (Union[float, str]): Stride to extract the start times
            of the clips. Integer or real values give explicit stride, in
            seconds. If string, must be one of the following:
                - "start": extract one clip per segment, starting at
                    onset/termination label.

        window_len (float): Lenght of windows to split the clip in in seconds.
            If negative no windowing is performed.

        fft_coeffs (Optional[List[Optional[int]]]): FFT coefficient interval
            *[min_index, max_index]*. Include all with ``[None]`` or switch off
            FFT with ``None``.

        node_level (bool): Wheter to work with node-level or global labels
    """

    diff_channels: bool
    sampling_rate: int  # Hz
    clip_length: float  # seconds
    clip_stride: Union[float, str]  # seconds
    window_len: float  # seconds

    # Interval [min, max] - include all with [null] or switch off with null
    fft_coeffs: Optional[List[Optional[int]]]

    node_level: bool


@dataclass
class DataConf:
    """Dataclass of data-related parameters

    Attributes:
        dataset (str): code of dataset to preprocess. Currently supported:
            - tusz: TUH Seizure Corpus
        raw_path (str): Root folder for raw data (downloads)
        processed_path (str): Root folder for preprocessed data

        labels (DataLabelsConf): Seizure labels specifications

        signals (DataSignalsConf): Options for signals and clips processing

        tusz (DataSourceConf): Dataset parameters for TUH Seizure Corpus
    """

    dataset: str

    raw_path: str
    processed_path: str

    tusz: DataSourceConf
    labels: DataLabelsConf

    signals: DataSignalsConf

"""Functions to parse annotations for CHB-MIT"""
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
from pyedflib import EdfReader

from seiz_eeg.constants import RE_CHANNELS, RE_MONTAGES
from seiz_eeg.preprocess.chbmit.constants import MIT2TUH, NULL_CHANNELS, TYPOS
from seiz_eeg.schemas import ClipsDF


def replace_all(text: str, mapping: Dict[str, str]) -> str:
    for key, val in mapping.items():
        text = text.replace(key, val)
    return text


@pa.check_types
def parse_patient(raw_path: Path, patient: str) -> DataFrame[ClipsDF]:
    """Parse summary file for :arg:`patient` and read metadata fro references edf files.

    Args:
        raw_path (Path): Path to raw CHB-MIT data and annotations
        patient (str): Patient name (chb\\d\\d)

    Raises:
        IOError: If a session annotation does not correspond to expected format

    Returns:
        DataFrame[ClipsDF]: Segements dataframe for target patient
    """
    summary_path = raw_path / f"{patient}/{patient}-summary.txt"

    sr_info, _ch_info, *seg_info = summary_path.read_text().split("\n\n")
    sampling_rate = int(re.search(r"(\d+) Hz", sr_info).group(1))

    # We have times but not dates, so we use relative times
    zero_start = None
    prev_end = 0

    segments = []

    # Filter text blocks to only contain sessions info
    # seg_info = [info for info in seg_info if info.startswith("File Name")]

    for info in seg_info:
        if not info.startswith("File Name"):
            continue

        file_name, sess = re.search(
            rf"File Name: ({patient}_?((?:\w_)?\d+\+?)\.edf)",
            info,
        ).group(1, 2)

        file_path = raw_path / patient / file_name

        edf_reader = EdfReader(str(file_path))
        date = edf_reader.getStartdatetime()
        channels = {
            montage
            for raw_montage in edf_reader.getSignalLabels()
            if (montage := replace_all(raw_montage, dict(**TYPOS, **MIT2TUH))) not in NULL_CHANNELS
            and re.fullmatch("|".join([RE_MONTAGES, RE_CHANNELS]), montage)
        }

        try:
            start_h, start_m, _start_s = map(
                int, re.search(r"File Start Time: (\d+):(\d\d):(\d\d)", info).groups()
            )
            end_h, end_m, _end_s = map(
                int, re.search(r"File End Time: (\d+):(\d\d):(\d\d)", info).groups()
            )
        except AttributeError as err:
            raise IOError(f"Error in session {sess}") from err

        # Convert times to seconds
        file_start = (start_h * 60 + start_m) * 60 + start_m
        file_end = (end_h * 60 + end_m) * 60 + end_m

        if zero_start is None:
            zero_start = file_start

        segment_counter = 0

        seiz_num = int(re.search(r"Number of Seizures in File: (\d+)", info).group(1))
        if seiz_num > 0:
            # Seizure 1 Start Time
            seiz_starts = re.findall(r"Seizure (?:\d+ )?Start Time: *(\d+)", info)
            seiz_ends = re.findall(r"Seizure (?:\d+ )?End Time: *(\d+)", info)
            assert len(seiz_starts) == len(seiz_ends) == seiz_num

            seiz_starts = map(int, seiz_starts)
            seiz_ends = map(int, seiz_ends)

            prev_end = 0

            for seiz_start, seiz_end in zip(seiz_starts, seiz_ends):
                segments.append(
                    {
                        "patient": patient,
                        "session": sess,
                        "segment": segment_counter,
                        "label": 0,
                        "start_time": prev_end,
                        "end_time": seiz_start,
                        "date": date,
                        "sampling_rate": sampling_rate,
                        "signals_path": str(file_path),
                        "channels": channels,
                    }
                )
                segment_counter += 1

                segments.append(
                    {
                        "patient": patient,
                        "session": sess,
                        "segment": segment_counter,
                        "label": 1,
                        "start_time": seiz_start,
                        "end_time": seiz_end,
                        "date": date,
                        "sampling_rate": sampling_rate,
                        "signals_path": str(file_path),
                        "channels": channels,
                    }
                )
                segment_counter += 1

                prev_end = seiz_end

            segments.append(
                {
                    "patient": patient,
                    "session": sess,
                    "segment": segment_counter,
                    "label": 0,
                    "start_time": prev_end,
                    "end_time": file_end - file_start,
                    "date": date,
                    "sampling_rate": sampling_rate,
                    "signals_path": str(file_path),
                    "channels": channels,
                }
            )

        else:
            # [session, segment, label, start_time, end_time, date, sampling_rate, signals_path]
            segments.append(
                {
                    "patient": patient,
                    "session": sess,
                    "segment": 0,
                    "label": 0,
                    "start_time": 0,
                    "end_time": file_end - file_start,
                    "date": date,
                    "sampling_rate": sampling_rate,
                    "signals_path": str(file_path),
                    "channels": channels,
                }
            )

    df = (
        pd.DataFrame(segments)
        .set_index([ClipsDF.patient, ClipsDF.session, ClipsDF.segment])
        .astype({ClipsDF.start_time: float, ClipsDF.end_time: float})
    )
    df[ClipsDF.date] = df[ClipsDF.date].dt.tz_localize(None)

    return df

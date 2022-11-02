import re
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import pandera as pa
from mne.io import read_raw_edf
from pandera.typing import DataFrame

from seiz_eeg.schemas import ClipsDF


@pa.check_types
def parse_patient(raw_path: Path, patient: str) -> DataFrame[ClipsDF]:
    summary_path = raw_path / f"{patient}/{patient}-summary.txt"

    sr_info, _ch_info, *seg_info = summary_path.read_text().split("\n\n")
    sampling_rate = int(re.search(r"(\d+) Hz", sr_info).group(1))

    # We have times but not dates, so we use relative times
    zero_start = None
    prev_end = 0

    segments = []

    for sess, info in enumerate(seg_info):
        file_name = re.search(r"File Name: (\w+\.edf)", info).group(1)
        file_path = raw_path / patient / file_name

        date = read_raw_edf(file_path, preload=False, verbose=False).info["meas_date"]

        try:
            start_h, start_m, start_s = map(
                int, re.search(r"File Start Time: (\d+):(\d\d):(\d\d)", info).groups()
            )
            end_h, end_m, end_s = map(
                int, re.search(r"File End Time: (\d+):(\d\d):(\d\d)", info).groups()
            )
        except AttributeError as err:
            raise IOError(f"Error in session {sess}") from err

        # Convert times to seconds
        file_start = (start_h * 60 + start_m) * 60 + start_s
        file_end = (end_h * 60 + end_m) * 60 + end_s

        if zero_start is None:
            zero_start = file_start

        segment_counter = 0

        seiz_num = int(re.search(r"Number of Seizures in File: (\d+)", info).group(1))
        if seiz_num > 0:
            seiz_starts = map(int, re.search(r"Seizure Start Time: (\d+)", info).groups())
            seiz_ends = map(int, re.search(r"Seizure End Time: (\d+)", info).groups())

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
                        "signals_path": file_path,
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
                        "signals_path": file_path,
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
                    "signals_path": file_path,
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
                    "signals_path": file_path,
                }
            )

    return (
        pd.DataFrame(segments)
        .set_index([ClipsDF.patient, ClipsDF.session, ClipsDF.segment])
        .astype({ClipsDF.start_time: float, ClipsDF.end_time: float})
    )

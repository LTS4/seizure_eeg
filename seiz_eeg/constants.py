"""Constants for different datasets"""

################################################################################
# OUTPUT FILENAMES

FILE_SEGMENTS_DF = "segments.parquet"
FILE_SIGNALS = "signals.parquet"

################################################################################
# CHANNEL NAMES

GLOBAL_CHANNEL = "global"


################################################################################
# SIGNALS CHANNELS/MONTAGES

_RE_CHANNEL = r"[FTPCO]+(?:\d+|Z)"
RE_CHANNELS = f"^{_RE_CHANNEL}$"
EEG_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
]

RE_MONTAGES = f"({_RE_CHANNEL})-({_RE_CHANNEL})"
EEG_MONTAGES = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]

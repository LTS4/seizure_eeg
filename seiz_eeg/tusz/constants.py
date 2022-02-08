"""Constants for TUSZ data retrieval"""

################################################################################
# OUTPUT FILENAMES

FILE_SEGMENTS_DF = "segments.parquet"
FILE_SIGNALS = "signals.parquet"

################################################################################
# CHANNEL NAMES

GLOBAL_CHANNEL = "global"

################################################################################
# FILE EXTENSIONS
SUFFIXES = [".edf", ".tse", ".lbl"]
SUFFIXES_BINARY = [".edf", ".tse_bi", ".lbl_bi"]

################################################################################
# SEIZURE TYPES

SEIZURE_VOC = {}

################################################################################
# REGULAR EXPRESSIONS

# ANNOTATION REGEXES
REGEX_LABEL = (
    r"\{(?P<level>\d+), (?P<unk>\d+), (?P<start>\d+\.\d{4}), "
    r"(?P<end>\d+\.\d{4}), (?P<montage_n>\d+), (?P<oh_sym>\[.*\])\}"
)
REGEX_SYMBOLS = r"symbols\[(?P<level>\d+)\].*(?P<sym_dict>\{.*\})"
REGEX_MONTAGE = r"(?P<num>\d+), (?P<montage>\w+\d*-\w+\d*):"

# SIGNALS REGEXES
SIGNAL_CHANNELS_FMT = r"EEG {}-\w+"
REGEX_SIGNAL_CHANNELS = SIGNAL_CHANNELS_FMT.format(r"(?P<ch>[\w\d]+)")

################################################################################
# SIGNALS CHANNELS/MONTAGES

CHANNELS = [
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


MONTAGES = [
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

################################################################################
# EXPECTED TYPOS

TYPOS = {
    "bkgd": "bckg",
}

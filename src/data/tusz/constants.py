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
TEMPLATE_SIGNAL_CHANNELS = "EEG {}-REF"
REGEX_SIGNAL_CHANNELS = TEMPLATE_SIGNAL_CHANNELS.format(r"(?P<ch>[\w\d]+)")


################################################################################
# EXPECTED TYPOS

TYPOS = {
    "bkgd": "bckg",
}
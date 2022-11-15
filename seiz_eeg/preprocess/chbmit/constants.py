"""CHB-MIT specific constants"""
NULL_CHANNELS = {"", "-", "."}
# MIT uses the 10-10 nomenclature while TUH uses the standard 10-20
MIT2TUH = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
TYPOS = {
    "01": "O1",
    "-CS2": "",
    "-Ref": "",
}

REPLACEMENTS = dict(**TYPOS, **MIT2TUH)

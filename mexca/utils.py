"""Utility functions.
"""

# Adapted from whisper.utils
# See: https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/utils.py

def str2bool(string: str):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string: str):
    return None if string == "None" else int(string)


def optional_float(string: str):
    return None if string == "None" else float(string)


def optional_str(string: str):
    return None if string == "None" else str(string)


def bool_or_str(string: str):
    try: 
        return str2bool(string)
    except ValueError:
        return string
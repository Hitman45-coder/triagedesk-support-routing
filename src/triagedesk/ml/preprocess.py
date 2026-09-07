import re
import unicodedata

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
LONG_NUMBER_RE = re.compile(r"(?<!\w)(?:\d[ -]?){8,}\d(?!\w)")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Apply the same conservative normalization during training and inference."""
    value = unicodedata.normalize("NFKC", value)
    value = EMAIL_RE.sub(" <EMAIL> ", value)
    value = LONG_NUMBER_RE.sub(" <NUMBER> ", value)
    return WHITESPACE_RE.sub(" ", value).strip()

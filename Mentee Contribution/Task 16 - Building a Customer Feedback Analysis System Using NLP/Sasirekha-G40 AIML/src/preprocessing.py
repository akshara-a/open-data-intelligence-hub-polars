import re
from html import unescape

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = unescape(str(text)).lower().strip()
    text = EMAIL_RE.sub(" email ", text)
    text = URL_RE.sub(" url ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()

def tokenize(text: str) -> list[str]:
    return clean_text(text).split()

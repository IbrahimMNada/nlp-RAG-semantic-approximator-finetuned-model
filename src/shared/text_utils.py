"""
Text normalization utilities for Arabic content.
Includes:
- Hidden Unicode / control character removal (RTL marks, zero-width chars)
- Arabic diacritics removal
- Letter normalization
- Digit normalization
- Whitespace normalization
"""

import re
import unicodedata
from typing import List


# ------------------------------------------------------------------
# Hidden / problematic Unicode characters (VERY IMPORTANT)
# ------------------------------------------------------------------

# RTL/LTR marks, zero-width chars, BOM, directional formatting
HIDDEN_UNICODE_RE = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]"
)


def clean_hidden_unicode(text: str) -> str:
    """
    Remove hidden Unicode control characters that commonly
    break tokenizers and embedding models (especially Arabic).
    """
    if not text:
        return ""

    # Normalize Unicode form (critical for Arabic consistency)
    text = unicodedata.normalize("NFKC", text)

    # Remove known hidden directional / zero-width chars
    text = HIDDEN_UNICODE_RE.sub("", text)

    # Remove other non-printable control characters
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
        or ch in ("\n", "\t")
    )

    return text


# ------------------------------------------------------------------
# Arabic-specific normalization
# ------------------------------------------------------------------

# Arabic diacritics (compiled once)
ARABIC_DIACRITICS = re.compile(r"""
    ّ    | # Shadda
    َ    | # Fatha
    ً    | # Tanwin Fath
    ُ    | # Damma
    ٌ    | # Tanwin Damm
    ِ    | # Kasra
    ٍ    | # Tanwin Kasr
    ْ    | # Sukun
    ـ      # Tatweel
""", re.VERBOSE)

# Arabic → Western digits
ARABIC_DIGIT_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789"
)

# Character normalization patterns
ALEF_VARIANTS = re.compile(r"[إأآا]")
YEH_VARIANT = re.compile(r"ى")
TEH_MARBUTA = re.compile(r"ة")
WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text by:
    1. Removing hidden Unicode control characters
    2. Normalizing Unicode form
    3. Converting Arabic numerals to English digits
    4. Removing diacritics
    5. Normalizing letter variants
    6. Normalizing whitespace
    """
    if not text:
        return ""

    # CRITICAL: clean hidden Unicode FIRST
    text = clean_hidden_unicode(text)

    # Normalize digits
    text = text.translate(ARABIC_DIGIT_MAP)

    # Normalize letters and diacritics
    text = ARABIC_DIACRITICS.sub("", text)
    text = ALEF_VARIANTS.sub("ا", text)
    text = YEH_VARIANT.sub("ي", text)
    text = TEH_MARBUTA.sub("ه", text)

    # Normalize whitespace
    text = WHITESPACE.sub(" ", text).strip()

    return text


def normalize_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Normalize a list of Arabic paragraphs safely.
    """
    return [normalize_arabic(p) for p in paragraphs if p]

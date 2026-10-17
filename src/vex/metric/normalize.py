"""Value normalization before comparison.

Gold values come from annotators (who may type "01/02/2025" or "ab 123 cd") and
predictions from the model, so both sides go through the same normalizer.
Each normalizer returns a canonical string, or `None` for empty input. A value
a specialised normalizer cannot parse (e.g. a date written as "début mars")
falls back to `normalize_text`, so it can still match an identical gold text
but never a real date.
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

MONTHS = {
    "janvier": 1, "janv": 1, "jan": 1,
    "fevrier": 2, "fevr": 2, "fev": 2,
    "mars": 3, "mar": 3,
    "avril": 4, "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7, "juil": 7,
    "aout": 8,
    "septembre": 9, "sept": 9, "sep": 9,
    "octobre": 10, "oct": 10,
    "novembre": 11, "nov": 11,
    "decembre": 12, "dec": 12,
}

_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.\-\s](\d{1,2})[/.\-\s](\d{2}|\d{4})$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_WORD_DATE = re.compile(r"^(\d{1,2})(?:er)?\s+([a-z]+)\.?\s+(\d{4})$")
_SIV = re.compile(r"^([A-Z]{2})(\d{3})([A-Z]{2})$")
_FNI = re.compile(r"^(\d{1,4})([A-Z]{2,3})(\d{2}|2A|2B|97\d)$")


def strip_accents(text: str) -> str:
    """Remove diacritics (é → e, ç → c) and fold ligatures (ﬀ → ff).

    Args:
        text: Any text.

    Returns:
        ASCII-folded text.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value: str | None) -> str | None:
    """Casefold, strip accents, turn punctuation into spaces, collapse whitespace.

    Args:
        value: Raw text.

    Returns:
        Canonical text, or `None` if nothing alphanumeric is left.
    """
    if value is None:
        return None
    text = strip_accents(value).casefold()
    text = re.sub(r"[^\w]+", " ", text).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_identifier(value: str | None) -> str | None:
    """Keep only letters and digits, uppercased (contract numbers, insurer codes).

    "AF-77-31-552", "af 77 31 552" and "AF7731552" all become "AF7731552".
    Leading zeros are kept: "0457" and "457" are different codes.

    Args:
        value: Raw identifier.

    Returns:
        Canonical identifier, or `None` if empty.
    """
    if value is None:
        return None
    text = re.sub(r"[^A-Za-z0-9]", "", strip_accents(value)).upper()
    return text or None


def _expand_year(year: str) -> int:
    number = int(year)
    if len(year) == 4:
        return number
    return 2000 + number if number < 70 else 1900 + number


def normalize_date(value: str | None) -> str | None:
    """Parse a French or ISO date into `YYYY-MM-DD`.

    Accepted: `2025-02-01`, `01/02/2025`, `1/2/25`, `01.02.2025`, `01-02-2025`,
    `01 02 2025`, `1er février 2025`, `1 fev. 2025`. Day comes first in
    numeric French dates. Two-digit years: 00-69 → 20xx, 70-99 → 19xx.

    Args:
        value: Raw date text.

    Returns:
        ISO date; the `normalize_text` form if unparseable; `None` if empty.
    """
    if value is None:
        return None
    text = re.sub(r"\s+", " ", strip_accents(value).casefold().strip())
    if not text:
        return None
    try:
        if match := _ISO_DATE.match(text):
            year, month, day = (int(group) for group in match.groups())
        elif match := _NUMERIC_DATE.match(text):
            day, month, year = int(match[1]), int(match[2]), _expand_year(match[3])
        elif (match := _WORD_DATE.match(text)) and match[2] in MONTHS:
            day, month, year = int(match[1]), MONTHS[match[2]], int(match[3])
        else:
            return normalize_text(value)
        return date(year, month, day).isoformat()
    except ValueError:
        return normalize_text(value)


def normalize_amount(value: str | None) -> str | None:
    """Parse an amount in euros into a canonical `1234.56` string.

    Accepted: `1 234,56 €`, `1234.56`, `1.234,56 EUR`, `300 €`, `1 234,5`.
    Not used by the current schema (decision §9-9), kept for amount fields.

    Args:
        value: Raw amount text.

    Returns:
        Amount with two decimals; the `normalize_text` form if unparseable;
        `None` if empty.
    """
    if value is None:
        return None
    text = re.sub(r"(?i)eur(os?)?|€|\s| | ", "", value)
    if not text:
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        return f"{Decimal(text):.2f}"
    except InvalidOperation:
        return normalize_text(value)


def normalize_plate(value: str | None) -> str | None:
    """Canonicalize a French registration plate.

    SIV (`AB-123-CD`, `ab 123 cd`, `AB123CD`) → `AB-123-CD`.
    FNI (`1234 AB 56`, `123-ABC-75`, `1234AB2A`) → `1234 AB 56`.
    Anything else → uppercase letters and digits only.

    Args:
        value: Raw plate text.

    Returns:
        Canonical plate, or `None` if empty.
    """
    compact = normalize_identifier(value)
    if compact is None:
        return None
    if match := _SIV.match(compact):
        return "-".join(match.groups())
    if match := _FNI.match(compact):
        return " ".join(match.groups())
    return compact


def normalize_list(values: list[str] | None) -> frozenset[str] | None:
    """Normalize a list of texts into a set (order and duplicates ignored).

    Args:
        values: Raw items.

    Returns:
        Set of normalized items, or `None` if no item survives.
    """
    if values is None:
        return None
    items = frozenset(item for item in (normalize_text(value) for value in values) if item)
    return items or None

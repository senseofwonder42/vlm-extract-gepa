"""Normalization table: dates, amounts, text, identifiers, plates, lists."""

import pytest

from vex.metric.normalize import (
    normalize_amount,
    normalize_date,
    normalize_identifier,
    normalize_list,
    normalize_plate,
    normalize_text,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2025-02-01", "2025-02-01"),
        ("2025-2-1", "2025-02-01"),
        ("01/02/2025", "2025-02-01"),
        ("1/2/2025", "2025-02-01"),
        ("01/02/25", "2025-02-01"),
        ("31/12/99", "1999-12-31"),
        ("01.02.2025", "2025-02-01"),
        ("01-02-2025", "2025-02-01"),
        ("01 02 2025", "2025-02-01"),
        ("1er février 2025", "2025-02-01"),
        ("1er fevrier 2025", "2025-02-01"),
        ("1 FÉVRIER 2025", "2025-02-01"),
        ("15 août 2026", "2026-08-15"),
        ("3 déc. 2025", "2025-12-03"),
        ("28 fév 2026", "2026-02-28"),
        ("  14 juillet 2025 ", "2025-07-14"),
        ("1er sept 2024", "2024-09-01"),
        ("31/02/2025", "31 02 2025"),  # impossible date: kept as text, never equals a real date
        ("début mars", "debut mars"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 234,56 €", "1234.56"),
        ("1234.56", "1234.56"),
        ("1.234,56 EUR", "1234.56"),
        ("300 €", "300.00"),
        ("1 234,5", "1234.50"),
        ("1 500 euros", "1500.00"),
        ("1.000.000", "1000000.00"),
        ("sans franchise", "sans franchise"),
        ("", None),
        (None, None),
    ],
)
def test_normalize_amount(raw, expected):
    assert normalize_amount(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mutuelle Imaginaire de l'Ouest", "mutuelle imaginaire de l ouest"),
        ("  CITROËN ", "citroen"),
        ("Citroen", "citroen"),
        ("12, rue des Lilas — 69003 Lyon", "12 rue des lilas 69003 lyon"),
        ("Le Goﬀ", "le goff"),
        ("Jean-Baptiste", "jean baptiste"),
        ("---", None),
        (None, None),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AF-77-31-552", "AF7731552"),
        ("af 77 31 552", "AF7731552"),
        ("MIO/AUTO/2025/00981", "MIOAUTO202500981"),
        ("0457", "0457"),
        (" - ", None),
        (None, None),
    ],
)
def test_normalize_identifier(raw, expected):
    assert normalize_identifier(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AB-123-CD", "AB-123-CD"),
        ("ab 123 cd", "AB-123-CD"),
        ("AB123CD", "AB-123-CD"),
        ("AB - 123 - CD", "AB-123-CD"),
        ("1234 AB 56", "1234 AB 56"),
        ("123-ABC-75", "123 ABC 75"),
        ("1234AB2A", "1234 AB 2A"),
        ("12 AB 974", "12 AB 974"),
        ("WW-123-AB", "WW-123-AB"),
        ("VF1RFB00X12345678", "VF1RFB00X12345678"),  # a VIN stays a VIN
        ("", None),
        (None, None),
    ],
)
def test_normalize_plate(raw, expected):
    assert normalize_plate(raw) == expected


def test_normalize_list_ignores_order_case_accents_and_duplicates():
    assert normalize_list(["Vol", "Bris de glace", "vol"]) == normalize_list(["BRIS DE GLACE", "vol"])
    assert normalize_list(["Responsabilité civile"]) == frozenset({"responsabilite civile"})
    assert normalize_list([" ", "--"]) is None
    assert normalize_list(None) is None

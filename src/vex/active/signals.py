"""Uncertainty signals for unreviewed documents, each in [0, 1] (higher = more uncertain).

(a) `pareto_disagreement`: the top-K Pareto candidates extract different values.
(b) `self_consistency`: N extractor samples at temperature > 0 disagree with each
    other, and the extractor disagrees with Haiku.
(c) `rules`: business rules fail (end date before start date, malformed plate,
    missing required field, implausible date).
(d) `verifier`: the diagnostic model cannot find (and quote) an extracted value
    in the document.

Disagreement is measured on normalized values (`normalize_field`), so "AB-123-CD"
and "ab 123 cd" agree. Field weights (`metric.weights`) make a disagreement on
the plate count more than one on the brand.
"""

from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, JsonValue

from vex.extract import build_messages
from vex.llm.client import OpenRouterClient
from vex.metric.compare import normalize_field
from vex.metric.normalize import normalize_plate
from vex.schema import FIELD_NAMES

Values = dict[str, JsonValue]
REQUIRED_FIELDS = ("assureur", "numero_contrat", "nom_souscripteur", "immatriculation",
                   "date_debut_validite", "date_fin_validite")


def _field_disagreement(values: list[JsonValue], field: str) -> float:
    """Share of samples that differ from the most common normalized value (0 = unanimous)."""
    if len(values) < 2:
        return 0.0
    counts = Counter(normalize_field(field, value) for value in values)
    return 1.0 - counts.most_common(1)[0][1] / len(values)


def disagreement(samples: list[Values | None], weights: dict[str, float]) -> float:
    """Weighted disagreement among several extractions of the same document.

    An unparseable sample (`None`) disagrees with everything: if any sample is
    `None`, every field is counted as disputed for that sample.

    Args:
        samples: Extractions of one document (two or more).
        weights: Field weights.

    Returns:
        Σ weight·field_disagreement / Σ weight, in [0, 1].
    """
    if len(samples) < 2:
        return 0.0
    total = sum(weights[field] for field in FIELD_NAMES)
    score = 0.0
    for field in FIELD_NAMES:
        values: list[JsonValue] = [s[field] if s is not None else f"<unparseable {i}>" for i, s in enumerate(samples)]
        score += weights[field] * _field_disagreement(values, field)
    return score / total


def pareto_disagreement(per_candidate: list[Values | None], weights: dict[str, float]) -> float:
    """(a) Disagreement among the top-K Pareto candidates' extractions.

    Args:
        per_candidate: One extraction per candidate.
        weights: Field weights.
    """
    return disagreement(per_candidate, weights)


def self_consistency(samples: list[Values | None], reference: Values | None, haiku: Values | None,
                     weights: dict[str, float]) -> float:
    """(b) Mean of sample disagreement and extractor-vs-Haiku disagreement.

    Args:
        samples: N extractor samples at temperature > 0.
        reference: Deterministic extraction (`extractor_eval`).
        haiku: The preannotator's extraction.
        weights: Field weights.
    """
    return (disagreement([reference, *samples], weights) + disagreement([reference, haiku], weights)) / 2


def rule_violations(values: Values | None, today: date) -> list[str]:
    """(c) Business-rule checks on one extraction.

    Rules: every required field present; ISO dates parse and lie within
    [today − 5 years, today + 2 years]; end date after start date; the plate
    is a valid SIV or FNI plate.

    Args:
        values: Extraction (`None` = unparseable: every rule fails).
        today: Reference date.

    Returns:
        Names of the failed checks.
    """
    if values is None:
        return ["unparseable"]
    failures = [f"missing:{field}" for field in REQUIRED_FIELDS if values.get(field) in (None, "", [])]
    dates: dict[str, date] = {}
    for field in ("date_debut_validite", "date_fin_validite"):
        raw = values.get(field)
        if isinstance(raw, str):
            try:
                dates[field] = date.fromisoformat(raw)
            except ValueError:
                failures.append(f"bad_date:{field}")
                continue
            if not today - timedelta(days=5 * 365) <= dates[field] <= today + timedelta(days=2 * 365):
                failures.append(f"implausible_date:{field}")
    if len(dates) == 2 and dates["date_fin_validite"] <= dates["date_debut_validite"]:
        failures.append("end_before_start")
    plate = values.get("immatriculation")
    if isinstance(plate, str):
        canonical = normalize_plate(plate) or ""
        if not ("-" in canonical or " " in canonical):  # neither SIV (AB-123-CD) nor FNI (1234 AB 56)
            failures.append("bad_plate")
    return failures


def rules_score(values: Values | None, today: date) -> float:
    """(c) Share of failed checks, in [0, 1].

    Args:
        values: Extraction (`None` = unparseable → 1.0, the most uncertain case).
        today: Reference date.
    """
    if values is None:
        return 1.0
    checks = len(REQUIRED_FIELDS) + 2 + 2 + 1 + 1  # required, 2 date parses, 2 ranges, order, plate
    return min(1.0, len(rule_violations(values, today)) / checks)


class FieldCheck(BaseModel):
    """Verifier verdict for one field."""

    field: str
    found: bool
    quote: str


class Verification(BaseModel):
    """Verifier answer for one document."""

    checks: list[FieldCheck]


VERIFIER_PROMPT = """Tu vérifies une extraction faite sur un justificatif d'assurance automobile français.
Pour chaque champ ci-dessous, la valeur extraite est-elle bien présente dans le document (images) ?
- found=true seulement si tu peux citer le texte exact du document qui la contient (quote, obligatoire) ;
- found=false si la valeur n'apparaît pas, ou si elle appartient à un autre élément (autre véhicule,
  numéro de sociétaire, intermédiaire…). Mets alors quote="".

Valeurs extraites :
{values}

Réponds uniquement en JSON."""

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": list(FIELD_NAMES)},
                    "found": {"type": "boolean"},
                    "quote": {"type": "string"},
                },
                "required": ["field", "found", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["checks"],
    "additionalProperties": False,
}


async def verifier_score(client: OpenRouterClient, document: Path, values: Values | None,
                         weights: dict[str, float]) -> float:
    """(d) Weighted share of extracted values the diagnostic model cannot find and quote.

    One call per document covers every non-null field.

    Args:
        client: OpenRouter client (uses the `diagnostic` role).
        document: Document file.
        values: Deterministic extraction.
        weights: Field weights.

    Returns:
        Score in [0, 1]; 1.0 if the extraction or the verifier answer is unusable.
    """
    if values is None:
        return 1.0
    present = {field: values[field] for field in FIELD_NAMES if values.get(field) not in (None, "", [])}
    if not present:
        return 0.0
    render = client.config.render
    messages = build_messages(VERIFIER_PROMPT.format(values=present), document, render.dpi, render.max_pages)
    result = await client.complete_json("diagnostic", messages, Verification, VERIFIER_SCHEMA,
                                        schema_name="verification")
    if result.parsed is None:
        return 1.0
    verified = {c.field for c in result.parsed.checks if c.found and c.quote.strip()}
    total = sum(weights[field] for field in present)
    return sum(weights[field] for field in present if field not in verified) / total


def combine(signals: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted mean of the available signals (a missing signal is left out, not counted as 0).

    Args:
        signals: Signal name → value, or `None` if not computed for this document.
        weights: `active.weights`.

    Returns:
        Combined uncertainty in [0, 1].
    """
    available = {name: value for name, value in signals.items() if value is not None}
    total = sum(weights[name] for name in available)
    return sum(weights[name] * value for name, value in available.items()) / total if total else 0.0

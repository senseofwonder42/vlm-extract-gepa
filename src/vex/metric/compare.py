"""Compare one predicted document with its gold values, field by field.

Rules (docs/PLAN.md §3, P3):

- each field has a *kind* that picks its normalizer (`FIELD_KINDS`);
- default comparison is exact equality of normalized values;
- fields listed in `metric.fuzzy_fields` match when the best of two rapidfuzz
  scores on normalized text reaches `metric.fuzzy_threshold`: `token_sort_ratio`
  (word order) and `ratio` with spaces removed (so "Pottiernec" vs "Pottier
  nec" is not an error). Both are deterministic;
- `garanties` is scored by set F1 (partial credit);
- error types: `omission` (gold present, prediction null), `hallucination`
  (gold null, prediction present), `wrong_value` (both present, different),
  `correct` (equal, including both null);
- a prediction that could not be parsed (`pred=None` for the whole document)
  types EVERY field as `omission` and scores 0 (decision §9-10).

Each field result carries a one-line diff in French and English. That line
is the textual feedback GEPA's reflection model reads, so it must say what was
expected, what was produced, and which error it is.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue
from rapidfuzz import fuzz

from vex.config import MetricConfig
from vex.metric.normalize import (
    normalize_date,
    normalize_identifier,
    normalize_list,
    normalize_plate,
    normalize_text,
)
from vex.schema import FIELD_NAMES

ErrorType = Literal["correct", "omission", "hallucination", "wrong_value"]
FieldKind = Literal["text", "identifier", "date", "plate", "list"]

FIELD_KINDS: dict[str, FieldKind] = {
    "assureur": "text",
    "numero_contrat": "identifier",
    "nom_souscripteur": "text",
    "adresse_souscripteur": "text",
    "immatriculation": "plate",
    "marque_vehicule": "text",
    "date_debut_validite": "date",
    "date_fin_validite": "date",
    "code_assureur": "identifier",
    "garanties": "list",
}

_LABELS: dict[ErrorType, str] = {
    "correct": "correct",
    "omission": "omission (oubli)",
    "hallucination": "hallucination (valeur inventée)",
    "wrong_value": "wrong_value (valeur erronée)",
}


@dataclass(frozen=True)
class FieldResult:
    """Comparison of one field.

    Attributes:
        field: Field name.
        gold: Gold value as annotated.
        pred: Predicted value as produced.
        score: 1.0 / 0.0, or the set F1 for list fields.
        error: Error type.
        diff: One-line French/English feedback.
    """

    field: str
    gold: JsonValue
    pred: JsonValue
    score: float
    error: ErrorType
    diff: str


def _is_empty(value: JsonValue) -> bool:
    return value is None or value == "" or value == []


def _normalize(kind: FieldKind, value: JsonValue) -> str | frozenset[str] | None:
    if _is_empty(value):
        return None
    if kind == "list":
        items = value if isinstance(value, list) else [value]
        return normalize_list([str(item) for item in items])
    text = str(value)
    if kind == "date":
        return normalize_date(text)
    if kind == "plate":
        return normalize_plate(text)
    if kind == "identifier":
        return normalize_identifier(text)
    return normalize_text(text)


def set_f1(gold: frozenset[str], pred: frozenset[str]) -> float:
    """F1 between two non-empty sets.

    Args:
        gold: Expected items.
        pred: Predicted items.

    Returns:
        2·|gold ∩ pred| / (|gold| + |pred|).
    """
    return 2 * len(gold & pred) / (len(gold) + len(pred))


def fuzzy_similarity(gold: str, pred: str) -> float:
    """Similarity (0-100) between two normalized texts, tolerant to word order and spacing.

    Args:
        gold: Normalized gold text.
        pred: Normalized predicted text.

    Returns:
        max(token_sort_ratio, ratio of the texts with spaces removed).
    """
    return max(fuzz.token_sort_ratio(gold, pred), fuzz.ratio(gold.replace(" ", ""), pred.replace(" ", "")))


def _fmt(value: JsonValue) -> str:
    return "null" if _is_empty(value) else repr(value)


def compare_field(field: str, gold: JsonValue, pred: JsonValue, config: MetricConfig) -> FieldResult:
    """Compare one field.

    Args:
        field: Field name (must be in `FIELD_KINDS`).
        gold: Gold value.
        pred: Predicted value.
        config: Metric settings (fuzzy fields and threshold).

    Returns:
        The field result.
    """
    kind = FIELD_KINDS[field]
    gold_norm, pred_norm = _normalize(kind, gold), _normalize(kind, pred)

    if gold_norm is None and pred_norm is None:
        error, score = "correct", 1.0
    elif pred_norm is None:
        error, score = "omission", 0.0
    elif gold_norm is None:
        error, score = "hallucination", 0.0
    elif kind == "list":
        score = set_f1(gold_norm, pred_norm)  # type: ignore[arg-type]
        error = "correct" if score == 1.0 else "wrong_value"
    elif field in config.fuzzy_fields:
        similar = fuzzy_similarity(gold_norm, pred_norm) >= config.fuzzy_threshold  # type: ignore[arg-type]
        error, score = ("correct", 1.0) if similar else ("wrong_value", 0.0)
    else:
        error, score = ("correct", 1.0) if gold_norm == pred_norm else ("wrong_value", 0.0)

    diff = f"[{_LABELS[error]}] {field}: attendu/expected {_fmt(gold)} — obtenu/got {_fmt(pred)}"
    if kind == "list" and error == "wrong_value":
        missing = sorted(gold_norm - pred_norm)  # type: ignore[operator]
        extra = sorted(pred_norm - gold_norm)  # type: ignore[operator]
        diff += f" (F1={score:.2f}; manquant/missing {missing}; en trop/extra {extra})"
    return FieldResult(field=field, gold=gold, pred=pred, score=score, error=error, diff=diff)


@dataclass(frozen=True)
class DocumentScore:
    """Comparison of one document.

    Attributes:
        doc_id: Document id.
        score: Weighted mean of field scores (0 if parsing failed).
        fields: Per-field results, in schema order.
        parse_failed: True if the model output could not be parsed.
    """

    doc_id: str
    score: float
    fields: list[FieldResult]
    parse_failed: bool

    def feedback(self) -> str:
        """Return the diff lines of every non-correct field (empty if perfect)."""
        return "\n".join(result.diff for result in self.fields if result.error != "correct")


def score_document(
    doc_id: str, gold: Mapping[str, JsonValue], pred: Mapping[str, JsonValue] | None, config: MetricConfig
) -> DocumentScore:
    """Score a whole document.

    Args:
        doc_id: Document id.
        gold: Gold values for every schema field.
        pred: Predicted values, or `None` if the model output could not be parsed.
        config: Metric settings (weights, fuzzy matching).

    Returns:
        The document score.

    Raises:
        KeyError: If a schema field has no weight in `config.weights`.
    """
    if pred is None:
        fields = [
            FieldResult(name, gold.get(name), None, 0.0, "omission",
                        f"[{_LABELS['omission']}] {name}: réponse illisible/unparseable model output")
            for name in FIELD_NAMES
        ]
        return DocumentScore(doc_id=doc_id, score=0.0, fields=fields, parse_failed=True)

    fields = [compare_field(name, gold.get(name), pred.get(name), config) for name in FIELD_NAMES]
    total_weight = sum(config.weights[name] for name in FIELD_NAMES)
    score = sum(config.weights[result.field] * result.score for result in fields) / total_weight
    return DocumentScore(doc_id=doc_id, score=score, fields=fields, parse_failed=False)

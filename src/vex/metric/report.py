"""Aggregate document scores into a report: global, per field, per error type."""

from dataclasses import dataclass

from vex.metric.compare import DocumentScore
from vex.schema import FIELD_NAMES

ERROR_TYPES = ("correct", "omission", "hallucination", "wrong_value")


@dataclass(frozen=True)
class Report:
    """Scores over a set of documents.

    Attributes:
        n_documents: Number of documents scored.
        global_score: Mean of the weighted document scores.
        parse_failures: Documents whose output could not be parsed.
        field_scores: Mean score per field.
        errors: Count per field and error type.
    """

    n_documents: int
    global_score: float
    parse_failures: int
    field_scores: dict[str, float]
    errors: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "n_documents": self.n_documents,
            "global_score": self.global_score,
            "parse_failures": self.parse_failures,
            "field_scores": self.field_scores,
            "errors": self.errors,
        }

    def table(self) -> str:
        """Return a fixed-width text table: one row per field plus a total row."""
        header = f"{'field':<22}{'score':>7}" + "".join(f"{e:>15}" for e in ERROR_TYPES)
        rows = [header, "-" * len(header)]
        for field in FIELD_NAMES:
            counts = self.errors[field]
            rows.append(f"{field:<22}{self.field_scores[field]:>7.3f}" + "".join(f"{counts[e]:>15}" for e in ERROR_TYPES))
        totals = {e: sum(self.errors[f][e] for f in FIELD_NAMES) for e in ERROR_TYPES}
        rows.append("-" * len(header))
        rows.append(f"{'GLOBAL (weighted)':<22}{self.global_score:>7.3f}" + "".join(f"{totals[e]:>15}" for e in ERROR_TYPES))
        rows.append(f"{self.n_documents} document(s), {self.parse_failures} parse failure(s)")
        return "\n".join(rows)


def build_report(scores: list[DocumentScore]) -> Report:
    """Aggregate document scores.

    Args:
        scores: One `DocumentScore` per document (may be empty).

    Returns:
        The report; all scores are 0 when `scores` is empty.
    """
    n = len(scores)
    field_scores = {
        field: (sum(r.score for s in scores for r in s.fields if r.field == field) / n if n else 0.0)
        for field in FIELD_NAMES
    }
    errors = {field: dict.fromkeys(ERROR_TYPES, 0) for field in FIELD_NAMES}
    for s in scores:
        for r in s.fields:
            errors[r.field][r.error] += 1
    return Report(
        n_documents=n,
        global_score=sum(s.score for s in scores) / n if n else 0.0,
        parse_failures=sum(s.parse_failed for s in scores),
        field_scores=field_scores,
        errors=errors,
    )

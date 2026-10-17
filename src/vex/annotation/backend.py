"""The `AnnotationBackend` interface and the backend factory.

A backend moves documents to human annotators and brings their work back:

- `push(items, round)`: send a batch, each item either `pre_annotated` (with
  the model's values shown as a starting point) or `blind` (no values at all);
- `pull(round)`: return the finished human annotations as store records,
  `reviewed` for pre-annotated items and `blind` for blind ones.

Writing records to the store is the caller's job, so both backends stay
transport-only. Selected by `annotation.backend: local | kili`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import JsonValue

from vex.annotation.store import AnnotationRecord
from vex.config import Config, Settings
from vex.metric.normalize import normalize_date
from vex.schema import AttestationAuto


@dataclass(frozen=True)
class PushItem:
    """One document sent for annotation.

    Attributes:
        doc_id: Document id.
        path: Document file.
        origin: `real` or `synthetic` (shown as metadata).
        status: `pre_annotated` (values shown) or `blind` (nothing shown).
        values: Proposed values; must be `None` for blind items.
    """

    doc_id: str
    path: Path
    origin: str
    status: Literal["pre_annotated", "blind"]
    values: dict[str, JsonValue] | None

    def __post_init__(self) -> None:
        """Blind items must not carry any proposal."""
        if self.status == "blind" and self.values is not None:
            raise ValueError(f"{self.doc_id}: a blind item cannot carry predicted values")


class AnnotationBackend(Protocol):
    """Transport between the pipeline and human annotators."""

    def push(self, items: list[PushItem], round_number: int) -> None:
        """Send a batch to annotators."""
        ...

    def pull(self, round_number: int) -> list[AnnotationRecord]:
        """Return the human annotations finished since the last pull."""
        ...


def clean_human_values(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Prepare human-typed values for the store.

    Dates may be typed the French way ("01/02/2025", "1er février 2025"):
    they are converted to ISO. A string in `garanties` is split on new lines
    and semicolons. The result is validated against `AttestationAuto`.

    Args:
        values: Raw values as typed by the annotator.

    Returns:
        Schema-valid values.

    Raises:
        pydantic.ValidationError: If a value still does not fit the schema
            (for instance a date that cannot be parsed).
    """
    cleaned = dict(values)
    for field in ("date_debut_validite", "date_fin_validite"):
        if isinstance(cleaned.get(field), str):
            cleaned[field] = normalize_date(cleaned[field])
    if isinstance(cleaned.get("garanties"), str):
        text = str(cleaned["garanties"]).replace(";", "\n")
        cleaned["garanties"] = [line.strip() for line in text.splitlines() if line.strip()]
    return AttestationAuto.model_validate(cleaned).model_dump(mode="json")


def make_backend(config: Config, settings: Settings) -> AnnotationBackend:
    """Build the backend selected by `annotation.backend`.

    Args:
        config: Resolved configuration.
        settings: Secrets (Kili credentials).

    Returns:
        A `LocalBackend` or a `KiliBackend`.
    """
    if config.annotation.backend == "kili":
        from vex.annotation.kili_backend import KiliBackend  # imported lazily: needs credentials

        return KiliBackend(config, settings)
    from vex.annotation.local import LocalBackend

    return LocalBackend(config.paths.inbox)

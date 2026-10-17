"""Append-only annotation store: `data/annotations/round_<N>.jsonl`.

One record per (document, round, step). A document typically gets a
`pre_annotated` record (the model's proposal) and then a `reviewed` record
(the human-corrected values), or a single `blind` record for held-out
documents annotated without any proposal.

Gold = the LATEST record of a document whose status is `reviewed` or `blind`.
A `pre_annotated` record is never gold, whatever its age. A later `reviewed`
record supersedes an earlier one, which is how a suspect annotation gets
corrected.

The effective GEPA sets are computed here from the frozen buckets of
`data/splits.yaml`: `gepa_train = bucket ∩ gold` and `gepa_pareto = bucket ∩ gold`.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

from vex.data.splits import Splits
from vex.schema import AttestationAuto

Status = Literal["pre_annotated", "reviewed", "blind"]
GOLD_STATUSES: tuple[Status, ...] = ("reviewed", "blind")
FORBIDDEN_ANNOTATORS = ("synthetic_meta",)


def now_iso() -> str:
    """Current UTC time, ISO 8601, second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class AnnotationRecord(BaseModel):
    """One annotation event.

    Attributes:
        record_id: Unique id of this record.
        doc_id: Document (sha256 of the file).
        round: Annotation round (0 = blind held-out batch).
        status: `pre_annotated`, `reviewed` or `blind`.
        values: Field values (`AttestationAuto` fields, null when absent).
        source: `model` for proposals, `human` for reviews.
        annotator: `model:<id>`, `local:<name>` or `kili:<email>`.
        review_seconds: Time spent reviewing, if known.
        comment: Free comment from the annotator (ambiguities, doubts).
        candidate_hash: Prompt used for the proposal (null for blind/human-only).
        model_id: Model that produced the proposal (null for human records).
        backend: `local` or `kili`.
        created_at: When the record was written.
        reviewed_at: When the human finished (human records only).
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_id: str
    round: int
    status: Status
    values: dict[str, JsonValue]
    source: Literal["model", "human"]
    annotator: str
    review_seconds: float | None = None
    comment: str | None = None
    candidate_hash: str | None = None
    model_id: str | None = None
    backend: Literal["local", "kili"]
    created_at: str = Field(default_factory=now_iso)
    reviewed_at: str | None = None

    @field_validator("values")
    @classmethod
    def _values_match_schema(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Values must validate against `AttestationAuto` (empty strings become null)."""
        return AttestationAuto.model_validate(values).model_dump(mode="json")

    @field_validator("annotator")
    @classmethod
    def _no_generator_labels(cls, annotator: str) -> str:
        """Refuse generator values posing as annotations (they are never labels)."""
        if any(forbidden in annotator for forbidden in FORBIDDEN_ANNOTATORS):
            raise ValueError(f"annotator {annotator!r} is not allowed: generator values are never labels")
        return annotator


class AnnotationStore:
    """Read and append annotation records."""

    def __init__(self, directory: Path) -> None:
        """Open the store.

        Args:
            directory: `data/annotations`.
        """
        self.directory = directory

    def append(self, records: list[AnnotationRecord]) -> None:
        """Append records to their round files.

        Args:
            records: Records to write.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            with (self.directory / f"round_{record.round}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")

    def append_new(self, records: list[AnnotationRecord]) -> list[AnnotationRecord]:
        """Append only records not already stored (backends may return the same work twice).

        Two records are the same work if they share doc_id, round, status, source,
        reviewed_at and values.

        Args:
            records: Candidate records.

        Returns:
            The records actually appended.
        """
        def key(record: AnnotationRecord) -> tuple:
            return (record.doc_id, record.round, record.status, record.source, record.reviewed_at,
                    tuple(sorted((k, str(v)) for k, v in record.values.items())))

        seen = {key(record) for record in self.records()}
        new = []
        for record in records:
            if key(record) not in seen:
                seen.add(key(record))
                new.append(record)
        self.append(new)
        return new

    def records(self) -> list[AnnotationRecord]:
        """Return every record, ordered by round file then line."""
        if not self.directory.exists():
            return []
        paths = sorted(self.directory.glob("round_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
        return [
            AnnotationRecord.model_validate_json(line)
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def gold(self) -> dict[str, AnnotationRecord]:
        """Return the gold record of every document that has one.

        "Latest" means the highest round, then the latest `created_at`.

        Returns:
            Mapping doc_id → latest `reviewed` or `blind` record.
        """
        gold: dict[str, AnnotationRecord] = {}
        for record in self.records():
            if record.status not in GOLD_STATUSES:
                continue
            current = gold.get(record.doc_id)
            if current is None or (record.round, record.created_at) >= (current.round, current.created_at):
                gold[record.doc_id] = record
        return gold

    def round_records(self, round_number: int) -> list[AnnotationRecord]:
        """Return the records of one round.

        Args:
            round_number: Round number.
        """
        return [record for record in self.records() if record.round == round_number]

    def annotated_doc_ids(self) -> set[str]:
        """Return every document that has any record (proposal or review)."""
        return {record.doc_id for record in self.records()}


def effective_split(name: Literal["gepa_train", "gepa_pareto"], splits: Splits, store: AnnotationStore) -> list[str]:
    """Return the documents of a GEPA bucket that have gold, in bucket order.

    Args:
        name: Bucket name. Held-out splits are deliberately not accepted here.
        splits: Frozen splits.
        store: Annotation store.

    Returns:
        doc_ids with a `reviewed` or `blind` record.

    Raises:
        ValueError: If `name` is not a GEPA bucket.
    """
    if name not in ("gepa_train", "gepa_pareto"):
        raise ValueError(f"{name!r} is not a GEPA bucket; held-out splits are never served here")
    gold = store.gold()
    return [doc_id for doc_id in getattr(splits, name) if doc_id in gold]


def held_out_ids(splits: Splits) -> set[str]:
    """Every document that must never reach GEPA, the reflection model or the rewriter.

    Args:
        splits: Frozen splits.

    Returns:
        heldout_blind ∪ heldout_standin ∪ reserved_unused.
    """
    return {*splits.heldout_blind, *splits.heldout_standin, *splits.reserved_unused}

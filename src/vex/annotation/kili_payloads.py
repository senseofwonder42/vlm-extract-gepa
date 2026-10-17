"""Pure builders for Kili payloads (SDK 2.142.1). No network, fully unit-tested.

Following template-kili (example 07, "un job TRANSCRIPTION par champ"):

- `json_interface`: one TRANSCRIPTION job per schema field, never required
  (an empty job means "absent" → null), plus an optional COMMENTAIRE job for
  ambiguities. The job instruction is the field's French description, so the
  annotators read the same rules as the model.
- `prediction_response`: the model's values as a `jsonResponse`. A null field
  is OMITTED (no pre-annotation shown) rather than sent as an empty string.
  `garanties` is one guarantee per line.
- `label_to_record`: a Kili label (as returned by `kili.labels(...)`) turned
  into a store record; empty text → null, `secondsToLabel` → review time.

Shapes were checked offline against the SDK's own parser (`ParsedJobs`, as
template-kili does); see tests/test_kili_payloads.py. They have NOT been
checked against a live Kili instance: see docs/kili_uncertainties.md.
"""

from typing import Literal

from pydantic import JsonValue

from vex.annotation.backend import clean_human_values
from vex.annotation.store import AnnotationRecord, now_iso
from vex.schema import FIELD_NAMES, field_descriptions

COMMENT_JOB = "COMMENTAIRE"
METADATA_KEYS = ("doc_id", "origin", "status", "round")


def job_name(field: str) -> str:
    """Kili job name of a schema field (e.g. `numero_contrat` → `NUMERO_CONTRAT`).

    Args:
        field: Schema field name.
    """
    return field.upper()


def _transcription_job(instruction: str) -> dict:
    # Same shape as template-kili's build_transcription_job(required=False).
    return {
        "mlTask": "TRANSCRIPTION",
        "content": {"categories": {}, "input": "textarea"},
        "instruction": instruction,
        "required": 0,
        "isChild": False,
    }


def json_interface(descriptions: dict[str, str] | None = None) -> dict:
    """Build the project's `json_interface`.

    Args:
        descriptions: Field instructions shown to annotators (default: the
            schema's seed descriptions).

    Returns:
        `{"jobs": {...}}`, one TRANSCRIPTION job per field plus COMMENTAIRE.
    """
    descriptions = descriptions or field_descriptions()
    jobs = {job_name(field): _transcription_job(f"{field} — {descriptions[field]}") for field in FIELD_NAMES}
    jobs[COMMENT_JOB] = _transcription_job(
        "Commentaire libre : ambiguïtés, doute sur un champ, plusieurs véhicules, document illisible…"
    )
    return {"jobs": jobs}


def _to_text(value: JsonValue) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def prediction_response(values: dict[str, JsonValue]) -> dict:
    """Turn model values into a Kili `jsonResponse` for `create_predictions`.

    Args:
        values: `AttestationAuto` values.

    Returns:
        `{JOB: {"text": "..."}}` for every non-null field.
    """
    response = {}
    for field in FIELD_NAMES:
        text = _to_text(values.get(field))
        if text is not None:
            response[job_name(field)] = {"text": text}
    return response


def asset_metadata(doc_id: str, origin: str, status: Literal["pre_annotated", "blind"], round_number: int) -> dict:
    """Asset `json_metadata` (filterable in Kili with `metadata_where`).

    Args:
        doc_id: Document id (also the asset's external_id).
        origin: `real` or `synthetic`.
        status: `pre_annotated` or `blind`.
        round_number: Annotation round.

    Returns:
        The metadata dict.
    """
    return {"doc_id": doc_id, "origin": origin, "status": status, "round": round_number}


def _job_text(json_response: dict, job: str) -> str | None:
    entry = json_response.get(job)
    if not isinstance(entry, dict):
        return None
    text = str(entry.get("text") or "").strip()
    return text or None


def label_to_record(label: dict, status: Literal["reviewed", "blind"], round_number: int) -> AnnotationRecord:
    """Convert one exported Kili label into a store record.

    Args:
        label: A label as returned by `kili.labels(fields=[...])`, with at least
            `jsonResponse`, `labelOf.externalId`, `author.email`, `secondsToLabel`
            and `createdAt`.
        status: `reviewed` (asset was pre-annotated) or `blind`.
        round_number: Annotation round.

    Returns:
        The store record.

    Raises:
        pydantic.ValidationError: If the annotated values do not fit the schema.
    """
    response = label.get("jsonResponse") or {}
    raw = {field: _job_text(response, job_name(field)) for field in FIELD_NAMES}
    seconds = label.get("secondsToLabel")
    return AnnotationRecord(
        doc_id=label["labelOf"]["externalId"],
        round=round_number,
        status=status,
        values=clean_human_values(raw),
        source="human",
        annotator=f"kili:{(label.get('author') or {}).get('email', 'unknown')}",
        review_seconds=float(seconds) if seconds is not None else None,
        comment=_job_text(response, COMMENT_JOB),
        backend="kili",
        reviewed_at=label.get("createdAt") or now_iso(),
    )


def latest_labels(labels: list[dict]) -> dict[str, dict]:
    """Keep one label per asset: REVIEW over DEFAULT, then the most recent.

    Args:
        labels: Labels of type DEFAULT or REVIEW.

    Returns:
        Mapping external_id → chosen label.
    """
    rank = {"REVIEW": 1, "DEFAULT": 0}
    chosen: dict[str, dict] = {}
    for label in labels:
        key = label["labelOf"]["externalId"]
        score = (rank.get(label.get("labelType", ""), -1), label.get("createdAt", ""))
        current = chosen.get(key)
        if current is None or score > (rank.get(current.get("labelType", ""), -1), current.get("createdAt", "")):
            chosen[key] = label
    return chosen

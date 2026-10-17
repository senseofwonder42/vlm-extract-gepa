"""Bootstrap rounds: build the first gold annotations and a better seed prompt.

Round N (micro-batches of 5 → 10 → 15 documents, `bootstrap.batch_sizes`):

1. select N documents from the GEPA buckets that have no annotation yet
   (real documents first), never from held-out lists;
2. pre-annotate them zero-shot with the `preannotator` role and the current
   candidate, store `pre_annotated` records, and push them to the backend;
3. (humans correct them)
4. pull the corrections; once the round is complete, gather every field the
   humans changed (rounds 1..N) with their comments;
5. the `reflection` model proposes rewritten `field:<name>` descriptions;
   the CLI shows a diff and saves the new candidate only after confirmation.

Held-out documents never reach steps 1-5: `held_out_ids` is checked at each
entry point.
"""

import asyncio
import difflib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, JsonValue

from vex.annotation.backend import PushItem
from vex.annotation.store import AnnotationRecord, AnnotationStore, held_out_ids
from vex.data.registry import DocRecord
from vex.config import MetricConfig
from vex.data.splits import Splits
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.metric.compare import compare_field
from vex.prompt import Candidate, candidate_hash, component_key
from vex.schema import FIELD_NAMES

REWRITE_INSTRUCTIONS = """You improve the field descriptions of a prompt that extracts fields from French motor \
insurance certificates (attestations, Mémo Véhicule Assuré, international/green cards).

Below are the current descriptions, then the corrections human annotators made to the model's output \
(predicted value → corrected value, plus their comments).

Rewrite ONLY the descriptions of fields whose corrections reveal a misunderstanding (where to look, \
what to exclude, format, when to answer null). Keep every rule that is still valid, stay concise, \
write in French like the originals, and keep the "null, jamais de chaîne vide" rule. Do not mention \
specific documents or values from the corrections. If no change is warranted, return an empty list."""


class FieldRewrite(BaseModel):
    """One proposed description."""

    field: str
    description: str
    rationale: str


class RewriteProposal(BaseModel):
    """The reflection model's answer."""

    rewrites: list[FieldRewrite]


REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewrites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": list(FIELD_NAMES)},
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["field", "description", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rewrites"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Correction:
    """A field the human changed.

    Attributes:
        doc_id: Document.
        field: Field name.
        predicted: Model value.
        corrected: Human value.
        comment: Annotator comment on the document (may be empty).
    """

    doc_id: str
    field: str
    predicted: JsonValue
    corrected: JsonValue
    comment: str | None


def _assert_not_held_out(doc_ids: set[str], splits: Splits) -> None:
    leaked = doc_ids & held_out_ids(splits)
    if leaked:
        raise RuntimeError(f"Held-out documents reached the bootstrap: {sorted(leaked)[:3]}...")


def select_batch(
    round_number: int, size: int, registry: list[DocRecord], splits: Splits, store: AnnotationStore, seed: int
) -> list[DocRecord]:
    """Pick the documents of a bootstrap round.

    Args:
        round_number: Round number (1-based).
        size: Batch size.
        registry: Registered documents.
        splits: Frozen splits.
        store: Annotation store (already annotated documents are skipped).
        seed: Selection seed.

    Returns:
        Up to `size` documents: unannotated real ones first, then seeded random synthetic ones.
    """
    buckets = set(splits.gepa_train) | set(splits.gepa_pareto)
    taken = store.annotated_doc_ids() | held_out_ids(splits)
    pool = [r for r in registry if r.doc_id in buckets and r.doc_id not in taken]
    real = [r for r in pool if r.origin == "real"]
    synthetic = [r for r in pool if r.origin == "synthetic"]
    random.Random(f"{seed}:bootstrap:{round_number}").shuffle(synthetic)
    batch = (real + synthetic)[:size]
    _assert_not_held_out({r.doc_id for r in batch}, splits)
    return batch


async def pre_annotate(
    client: OpenRouterClient,
    documents: list[DocRecord],
    candidate: Candidate,
    round_number: int,
    backend_name: Literal["local", "kili"],
) -> tuple[list[AnnotationRecord], list[PushItem]]:
    """Run the `preannotator` on a batch and build store records and push items.

    A document whose output could not be parsed is still pushed, with all
    fields null, so the annotator fills it in from scratch.

    Args:
        client: OpenRouter client.
        documents: Batch to pre-annotate.
        candidate: Candidate prompt.
        round_number: Round number.
        backend_name: `local` or `kili` (recorded in the store).

    Returns:
        `pre_annotated` records and the matching push items.
    """
    results = await asyncio.gather(
        *(extract_document(client, Path(d.path), candidate, role="preannotator") for d in documents)
    )
    model_id = client.config.role("preannotator").model
    records, items = [], []
    for document, result in zip(documents, results, strict=True):
        values = result.values or dict.fromkeys(FIELD_NAMES)
        if result.failed:
            logger.warning("{}: pre-annotation failed ({}), pushed empty", document.doc_id[:12], result.error)
        records.append(AnnotationRecord(
            doc_id=document.doc_id, round=round_number, status="pre_annotated", values=values,
            source="model", annotator=f"model:{model_id}", candidate_hash=candidate_hash(candidate),
            model_id=model_id, backend=backend_name,
        ))
        items.append(PushItem(document.doc_id, Path(document.path), document.origin, "pre_annotated", values))
    return records, items


def collect_corrections(
    store: AnnotationStore, up_to_round: int, splits: Splits, metric: MetricConfig
) -> list[Correction]:
    """Every field humans changed in bootstrap rounds 1..`up_to_round`.

    Args:
        store: Annotation store.
        up_to_round: Last round to include.
        splits: Frozen splits (for the held-out guard).
        metric: `MetricConfig`, so "changed" means "not equal after normalization".

    Returns:
        One `Correction` per changed field.
    """
    records = [r for r in store.records() if 1 <= r.round <= up_to_round]
    proposals = {r.doc_id: r for r in records if r.status == "pre_annotated"}
    gold = store.gold()
    _assert_not_held_out(set(proposals), splits)
    corrections = []
    for doc_id, proposal in proposals.items():
        reviewed = gold.get(doc_id)
        if reviewed is None or reviewed.status != "reviewed":
            continue
        for field in FIELD_NAMES:
            result = compare_field(field, reviewed.values.get(field), proposal.values.get(field), metric)
            if result.error != "correct":
                corrections.append(Correction(doc_id, field, proposal.values.get(field),
                                              reviewed.values.get(field), reviewed.comment))
    return corrections


async def propose_rewrites(
    client: OpenRouterClient, candidate: Candidate, corrections: list[Correction]
) -> list[FieldRewrite]:
    """Ask the `reflection` model for new field descriptions.

    Args:
        client: OpenRouter client.
        candidate: Current candidate.
        corrections: Human corrections (held-out documents excluded upstream).

    Returns:
        Proposed rewrites (possibly empty).
    """
    current = {field: candidate[component_key(field)] for field in FIELD_NAMES}
    payload = {
        "current_descriptions": current,
        "corrections": [
            {"field": c.field, "predicted": c.predicted, "corrected": c.corrected, "annotator_comment": c.comment}
            for c in corrections
        ],
    }
    messages = [
        {"role": "system", "content": REWRITE_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)},
    ]
    result = await client.complete_json("reflection", messages, RewriteProposal, REWRITE_SCHEMA,
                                        schema_name="rewrite_proposal")
    if result.parsed is None:
        logger.error("Reflection model returned no usable proposal: {}", result.error)
        return []
    return result.parsed.rewrites


def apply_rewrites(candidate: Candidate, rewrites: list[FieldRewrite]) -> Candidate:
    """Return a new candidate with the proposed descriptions.

    Args:
        candidate: Current candidate.
        rewrites: Proposed rewrites.
    """
    updated = dict(candidate)
    for rewrite in rewrites:
        updated[component_key(rewrite.field)] = rewrite.description.strip()
    return updated


def candidate_diff(old: Candidate, new: Candidate) -> str:
    """Unified diff of the components that changed.

    Args:
        old: Current candidate.
        new: Proposed candidate.

    Returns:
        A readable diff, one block per changed component.
    """
    blocks = []
    for key in old:
        if old[key] != new[key]:
            lines = difflib.unified_diff(
                old[key].split(". "), new[key].split(". "), fromfile=f"{key} (current)", tofile=f"{key} (proposed)",
                lineterm="",
            )
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

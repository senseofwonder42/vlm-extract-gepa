"""Score a candidate on documents that have gold annotations."""

import asyncio
from pathlib import Path

from vex.annotation.store import AnnotationRecord
from vex.config import RoleName
from vex.data.registry import DocRecord
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.metric.compare import DocumentScore, score_document
from vex.prompt import Candidate


async def score_candidate(
    client: OpenRouterClient,
    candidate: Candidate,
    doc_ids: list[str],
    registry: dict[str, DocRecord],
    gold: dict[str, AnnotationRecord],
    role: RoleName = "extractor_eval",
) -> list[DocumentScore]:
    """Extract every document with `candidate` and score it against its gold record.

    Args:
        client: OpenRouter client.
        candidate: Candidate prompt.
        doc_ids: Documents to score; each must have a gold record.
        registry: doc_id → registry record (for the file path).
        gold: doc_id → gold record.
        role: Extraction role (deterministic `extractor_eval` by default).

    Returns:
        One score per document, in `doc_ids` order.

    Raises:
        KeyError: If a document has no gold record or is not registered.
    """
    results = await asyncio.gather(
        *(extract_document(client, Path(registry[doc_id].path), candidate, role=role) for doc_id in doc_ids)
    )
    metric = client.config.metric
    return [
        score_document(doc_id, gold[doc_id].values, result.values, metric)
        for doc_id, result in zip(doc_ids, results, strict=True)
    ]

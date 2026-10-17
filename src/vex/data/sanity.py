"""Sanity check: score a candidate on synthetic documents against synthetic_meta.

This is the ONLY reader of synthetic_meta outside tests (guarded by
`tests/test_synth.py`). Its result is printed and saved in a run directory for
a human to look at; it is never written to the annotation store, never used
as labels, and never fed to GEPA. It only samples documents from the
`gepa_train`/`gepa_pareto` buckets, so held-out and stand-in documents are
never touched.
"""

import asyncio
import json
import random
from pathlib import Path

from vex.config import RoleName
from vex.data.registry import DocRecord
from vex.data.splits import Splits
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.metric.compare import DocumentScore, score_document
from vex.prompt import Candidate


def eligible_documents(registry: list[DocRecord], splits: Splits) -> list[DocRecord]:
    """Synthetic documents of the GEPA buckets, in registry order.

    Args:
        registry: Registered documents.
        splits: Current splits.

    Returns:
        Documents allowed in a sanity run.
    """
    pool = set(splits.gepa_train) | set(splits.gepa_pareto)
    return [record for record in registry if record.origin == "synthetic" and record.doc_id in pool]


async def run_sanity(
    client: OpenRouterClient,
    candidate: Candidate,
    documents: list[DocRecord],
    meta_path: Path,
    n: int,
    seed: int,
    role: RoleName = "extractor_eval",
) -> list[DocumentScore]:
    """Extract `n` sampled documents and score them against synthetic_meta.

    Args:
        client: OpenRouter client.
        candidate: Candidate prompt.
        documents: Eligible documents (see `eligible_documents`).
        meta_path: synthetic_meta JSONL file.
        n: Number of documents to sample.
        seed: Sampling seed.
        role: Extraction role.

    Returns:
        One score per sampled document.
    """
    meta = {
        line["file"]: line["values"]
        for line in map(json.loads, meta_path.read_text(encoding="utf-8").splitlines())
    }
    sample = random.Random(seed).sample(documents, k=min(n, len(documents)))
    results = await asyncio.gather(
        *(extract_document(client, Path(record.path), candidate, role=role) for record in sample)
    )
    return [
        score_document(record.doc_id, meta[Path(record.path).name], result.values, client.config.metric)
        for record, result in zip(sample, results, strict=True)
    ]

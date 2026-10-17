"""`vex select`: pick the next documents to annotate, push them as round N+1.

1. Pool: unannotated documents of the GEPA buckets (never held-out lists),
   sampled down to `active.max_pool` to bound the cost.
2. Candidates: the top-K of the latest `vex optimize` run (by valset average),
   else the latest saved prompt. The first one is the "current best".
3. Signals per document (`signals.py`): (a) Pareto disagreement (needs ≥ 2
   candidates), (b) self-consistency, (c) rules, then (d) the verifier on the
   `active.verifier_shortlist` most uncertain documents only.
4. Batch: `active.batch_size` documents. Most are the most uncertain, picked
   across image clusters (`diversity.py`); a `random_confident_fraction` share
   is drawn at random from the confident half, to measure residual error
   (errors the signals did not see).
5. Push as `pre_annotated` round N+1, prefilled with the Haiku extraction
   already computed for signal (b).
"""

import asyncio
import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from loguru import logger

from vex.active.diversity import embed_documents, pick_diverse
from vex.active.signals import combine, pareto_disagreement, rules_score, self_consistency, verifier_score
from vex.annotation.backend import AnnotationBackend, PushItem
from vex.annotation.store import AnnotationRecord, AnnotationStore, held_out_ids
from vex.config import Config
from vex.data.registry import DocRecord
from vex.data.splits import Splits
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.prompt import Candidate, candidate_hash, load_candidate, seed_candidate
from vex.schema import FIELD_NAMES


@dataclass
class DocSignals:
    """Signals of one pool document.

    Attributes:
        doc_id: Document id.
        path: Document file.
        origin: `real` or `synthetic`.
        pareto: (a), `None` if fewer than 2 candidates.
        self_consistency: (b).
        rules: (c).
        verifier: (d), `None` if the document was not shortlisted.
        combined: Weighted combination of the available signals.
        haiku_values: Preannotator extraction (pushed as pre-annotation).
        reference_values: Deterministic extraction with the best candidate (checked by the verifier).
    """

    doc_id: str
    path: str
    origin: str
    pareto: float | None
    self_consistency: float
    rules: float
    verifier: float | None
    combined: float
    haiku_values: dict | None
    reference_values: dict | None

    def signal_dict(self) -> dict[str, float | None]:
        """Signals keyed by their `active.weights` names."""
        return {"pareto_disagreement": self.pareto, "self_consistency": self.self_consistency,
                "rules": self.rules, "verifier": self.verifier}


@dataclass(frozen=True)
class Selection:
    """One selected document and why.

    Attributes:
        doc_id: Document id.
        reason: `uncertain` or `random_confident`.
        cluster: Image cluster (uncertain picks only).
    """

    doc_id: str
    reason: str
    cluster: int | None


def current_candidates(config: Config, k: int) -> list[Candidate]:
    """Top-K candidates of the latest `vex optimize` run, else the latest saved prompt.

    Args:
        config: Resolved configuration.
        k: Number of candidates wanted.

    Returns:
        1 to `k` distinct candidates, best first.
    """
    runs = sorted(config.paths.runs.glob("*-optimize-*/candidates.jsonl"))
    if runs:
        entries = [json.loads(line) for line in runs[-1].read_text(encoding="utf-8").splitlines()]
        ranked, seen = [], set()
        for entry in sorted(entries, key=lambda e: -e["average_score"]):
            if entry["candidate_hash"] not in seen:
                seen.add(entry["candidate_hash"])
                ranked.append(entry["candidate"])
        if ranked:
            logger.info("Candidates: top {} of {}", min(k, len(ranked)), runs[-1].parent.name)
            return ranked[:k]
    saved = sorted(config.paths.prompts.glob("*.json"), key=lambda p: p.stat().st_mtime)
    preferred = [p for p in saved if p.name.startswith("gepa-best")] or [p for p in saved if p.name.startswith("bootstrap-r")]
    if preferred:
        logger.info("Candidate: {}", preferred[-1].name)
        return [load_candidate(preferred[-1])]
    return [seed_candidate()]


def build_pool(registry: list[DocRecord], splits: Splits, store: AnnotationStore, max_pool: int, seed: int) -> list[DocRecord]:
    """Unannotated documents of the GEPA buckets, sampled down to `max_pool`.

    Args:
        registry: Registered documents.
        splits: Frozen splits.
        store: Annotation store.
        max_pool: Maximum pool size.
        seed: Sampling seed.

    Returns:
        Pool documents (never held-out ones).
    """
    buckets = set(splits.gepa_train) | set(splits.gepa_pareto)
    excluded = held_out_ids(splits) | store.annotated_doc_ids()
    pool = [r for r in registry if r.doc_id in buckets and r.doc_id not in excluded]
    if len(pool) > max_pool:
        pool = random.Random(f"{seed}:pool").sample(pool, max_pool)
    return pool


async def compute_signals(client: OpenRouterClient, pool: list[DocRecord], candidates: list[Candidate],
                          config: Config, today: date) -> list[DocSignals]:
    """Compute (a)-(c) for every pool document, then (d) on the shortlist.

    Args:
        client: OpenRouter client.
        pool: Documents to score.
        candidates: Current candidates, best first.
        config: Resolved configuration.
        today: Reference date for the rules.

    Returns:
        One `DocSignals` per document, in pool order.
    """
    active, weights = config.active, config.metric.weights
    best = candidates[0]
    temperature = {"temperature": active.self_consistency.temperature}

    async def one(record: DocRecord) -> DocSignals:
        path = Path(record.path)
        reference = await extract_document(client, path, best, role="extractor_eval")
        others = await asyncio.gather(*(extract_document(client, path, c, role="extractor_eval") for c in candidates[1:]))
        samples = await asyncio.gather(*(
            extract_document(client, path, best, role="extractor", cache_salt=f"self-consistency-{i}",
                             extra_params=temperature)
            for i in range(active.self_consistency.n)))
        haiku = await extract_document(client, path, best, role="preannotator")
        pareto = pareto_disagreement([reference.values, *(o.values for o in others)], weights) if others else None
        signals = DocSignals(
            doc_id=record.doc_id, path=record.path, origin=record.origin, pareto=pareto,
            self_consistency=self_consistency([s.values for s in samples], reference.values, haiku.values, weights),
            rules=rules_score(reference.values, today), verifier=None, combined=0.0, haiku_values=haiku.values,
            reference_values=reference.values)
        signals.combined = combine(signals.signal_dict(), active.weights)
        return signals

    results = await asyncio.gather(*(one(record) for record in pool))
    shortlist = sorted(results, key=lambda s: -s.combined)[: active.verifier_shortlist]
    scores = await asyncio.gather(*(
        verifier_score(client, Path(s.path), s.reference_values, weights) for s in shortlist
    ))
    for signals, score in zip(shortlist, scores, strict=True):
        signals.verifier = score
        signals.combined = combine(signals.signal_dict(), active.weights)
    return results


def plan_batch(signals: list[DocSignals], config: Config, seed: int) -> list[Selection]:
    """Choose the batch: diverse uncertain picks plus a random confident sample.

    Args:
        signals: Pool signals.
        config: Resolved configuration.
        seed: Seed for clustering and random picks.

    Returns:
        The selections (at most `active.batch_size`).
    """
    active = config.active
    k = min(active.batch_size, len(signals))
    n_random = round(k * active.random_confident_fraction)
    scores = {s.doc_id: s.combined for s in signals}
    ranked = sorted(signals, key=lambda s: (-s.combined, s.doc_id))
    confident_half = {s.doc_id for s in ranked[len(ranked) // 2:]}

    candidates = [s for s in ranked if s.doc_id not in confident_half] or ranked
    if active.diversity.enabled and len(candidates) > 1:
        embeddings, used = embed_documents([Path(s.path) for s in candidates], active.diversity.embedder)
        logger.info("Diversity: {} embeddings, {} clusters max", used, active.diversity.n_clusters)
        picked = pick_diverse([s.doc_id for s in candidates], embeddings, scores, active.diversity.n_clusters,
                              k - n_random, seed)
    else:
        picked = [(s.doc_id, None) for s in candidates[: k - n_random]]
    selections = [Selection(doc_id, "uncertain", cluster) for doc_id, cluster in picked]

    chosen = {s.doc_id for s in selections}
    confident = sorted(d for d in confident_half if d not in chosen)
    for doc_id in random.Random(f"{seed}:confident").sample(confident, min(k - len(selections), len(confident))):
        selections.append(Selection(doc_id, "random_confident", None))
    return selections


def push_selection(selections: list[Selection], signals: list[DocSignals], round_number: int, best: Candidate,
                   store: AnnotationStore, backend: AnnotationBackend, config: Config) -> None:
    """Store `pre_annotated` records and push the batch.

    Args:
        selections: Chosen documents.
        signals: Pool signals (carry the Haiku pre-annotations).
        round_number: New round number.
        best: Candidate used for the pre-annotation.
        store: Annotation store.
        backend: Annotation backend.
        config: Resolved configuration.
    """
    by_id = {s.doc_id: s for s in signals}
    model_id = config.role("preannotator").model
    records, items = [], []
    for selection in selections:
        doc = by_id[selection.doc_id]
        values = doc.haiku_values or dict.fromkeys(FIELD_NAMES)
        records.append(AnnotationRecord(
            doc_id=doc.doc_id, round=round_number, status="pre_annotated", values=values, source="model",
            annotator=f"model:{model_id}", candidate_hash=candidate_hash(best), model_id=model_id,
            backend=config.annotation.backend, comment=f"selected: {selection.reason}"))
        items.append(PushItem(doc.doc_id, Path(doc.path), doc.origin, "pre_annotated", values))
    store.append(records)
    backend.push(items, round_number)


def selection_report(signals: list[DocSignals], selections: list[Selection]) -> list[dict]:
    """Per-document signals and selection reason, for `selection.json`.

    Args:
        signals: Pool signals.
        selections: Chosen documents.
    """
    reasons = {s.doc_id: asdict(s) for s in selections}
    rows = []
    for s in sorted(signals, key=lambda s: -s.combined):
        row = {k: v for k, v in asdict(s).items() if k not in ("haiku_values", "reference_values")}
        row["selected"] = reasons.get(s.doc_id)
        rows.append(row)
    return rows

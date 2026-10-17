"""Per-round tracking: `reports/rounds.csv` and plateau detection.

One row per annotation round (re-running `vex report` for a round replaces its
row). The held-out score comes from `vex.optimize.score_heldout`, read-only:
it is reported, never fed back. On the Mac, `heldout_blind` is empty, so the
row uses `heldout_standin` and says so in `heldout_split`.

Plateau: the held-out score gained less than `tracking.plateau_delta` in
each of the last `tracking.plateau_rounds` rounds.
"""

import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from vex.annotation.store import AnnotationStore
from vex.config import MetricConfig
from vex.metric.compare import compare_field
from vex.schema import FIELD_NAMES

COLUMNS = [
    "round", "date", "candidate_hash", "heldout_split", "heldout_n", "heldout_global",
    *(f"heldout_{field}" for field in FIELD_NAMES),
    "n_gold_train", "n_gold_pareto", "fields_corrected_share", "median_review_seconds",
    "suspect_annotations", "usd_round", "usd_cumulative", "plateau",
]


def fields_corrected_share(store: AnnotationStore, round_number: int, metric: MetricConfig) -> float | None:
    """Share of pre-annotated fields the annotators changed in a round.

    Args:
        store: Annotation store.
        round_number: Round.
        metric: Metric settings ("changed" = not equal after normalization).

    Returns:
        Changed fields / reviewed fields, or `None` if nothing was reviewed.
    """
    records = store.round_records(round_number)
    proposals = {r.doc_id: r for r in records if r.status == "pre_annotated"}
    reviews = {r.doc_id: r for r in records if r.status == "reviewed" and r.doc_id in proposals}
    if not reviews:
        return None
    changed = sum(
        compare_field(field, review.values.get(field), proposals[doc_id].values.get(field), metric).error != "correct"
        for doc_id, review in reviews.items() for field in FIELD_NAMES
    )
    return changed / (len(reviews) * len(FIELD_NAMES))


def median_review_seconds(store: AnnotationStore, round_number: int) -> float | None:
    """Median review time of the round's human records that report one.

    Args:
        store: Annotation store.
        round_number: Round.
    """
    times = [r.review_seconds for r in store.round_records(round_number)
             if r.source == "human" and r.review_seconds is not None]
    return statistics.median(times) if times else None


def suspect_count(review_queue: Path) -> int:
    """Number of suspect annotations flagged by the diagnostic model.

    Args:
        review_queue: `data/review_queue`.
    """
    path = review_queue / "suspect_annotations.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0


def cumulative_usd(runs_dir: Path) -> float:
    """Total spend of every run directory (`cost.json` files).

    Args:
        runs_dir: `runs/`.
    """
    return sum(json.loads(p.read_text(encoding="utf-8"))["total_usd"] for p in runs_dir.glob("*/cost.json"))


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read `rounds.csv` (empty if missing).

    Args:
        csv_path: `reports/rounds.csv`.
    """
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def detect_plateau(scores: list[float], delta: float, rounds: int) -> bool:
    """True if each of the last `rounds` gains is below `delta`.

    Args:
        scores: Held-out scores in round order.
        delta: Minimum gain that counts as progress.
        rounds: Number of consecutive small gains required.

    Returns:
        Whether progress has stalled (False until there is enough history).
    """
    if len(scores) < rounds + 1:
        return False
    recent = scores[-(rounds + 1):]
    return all(later - earlier < delta for earlier, later in zip(recent, recent[1:], strict=False))


def build_row(round_number: int, candidate_hash: str, heldout: dict, n_gold_train: int, n_gold_pareto: int,
              corrected: float | None, review_seconds: float | None, suspects: int, usd_total: float,
              previous: list[dict[str, str]], plateau_delta: float, plateau_rounds: int) -> dict[str, str]:
    """Assemble one `rounds.csv` row.

    Args:
        round_number: Round.
        candidate_hash: Hash of the scored candidate.
        heldout: Output of `score_heldout` with a single candidate named `current`.
        n_gold_train: Gold documents in `gepa_train`.
        n_gold_pareto: Gold documents in `gepa_pareto`.
        corrected: `fields_corrected_share`.
        review_seconds: `median_review_seconds`.
        suspects: `suspect_count`.
        usd_total: `cumulative_usd`.
        previous: Existing rows of other rounds, in round order.
        plateau_delta: `tracking.plateau_delta`.
        plateau_rounds: `tracking.plateau_rounds`.

    Returns:
        The row, as strings (empty string = not available).
    """
    split = "heldout_blind" if heldout["heldout_blind"]["n"] else "heldout_standin"
    report = heldout[split]["scores"].get("current")
    fmt = lambda value: "" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))  # noqa: E731
    row = {column: "" for column in COLUMNS}
    row.update(
        round=str(round_number), date=datetime.now(UTC).date().isoformat(), candidate_hash=candidate_hash[:8],
        heldout_split=split, heldout_n=str(heldout[split]["n"]),
        heldout_global=fmt(report["global_score"]) if report else "",
        n_gold_train=str(n_gold_train), n_gold_pareto=str(n_gold_pareto),
        fields_corrected_share=fmt(corrected), median_review_seconds=fmt(review_seconds),
        suspect_annotations=str(suspects), usd_cumulative=fmt(usd_total),
    )
    if report:
        for field in FIELD_NAMES:
            row[f"heldout_{field}"] = fmt(report["field_scores"][field])
    earlier = [float(r["usd_cumulative"]) for r in previous if r["usd_cumulative"]]
    row["usd_round"] = fmt(usd_total - earlier[-1]) if earlier else fmt(usd_total)
    history = [float(r["heldout_global"]) for r in previous if r["heldout_global"]]
    if row["heldout_global"]:
        history.append(float(row["heldout_global"]))
    row["plateau"] = str(detect_plateau(history, plateau_delta, plateau_rounds))
    return row


def upsert_row(csv_path: Path, row: dict[str, str]) -> list[dict[str, str]]:
    """Write `row`, replacing any existing row of the same round; keep rounds sorted.

    Args:
        csv_path: `reports/rounds.csv`.
        row: Row to write.

    Returns:
        All rows after the update.
    """
    rows = [r for r in load_rows(csv_path) if r["round"] != row["round"]] + [row]
    rows.sort(key=lambda r: int(r["round"]))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows

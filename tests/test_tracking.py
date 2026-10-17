"""Per-round tracking: corrected share, review time, CSV upsert, plateau detection."""

import json

import pytest

from vex.annotation.store import AnnotationRecord, AnnotationStore
from vex.tracking import (
    COLUMNS,
    build_row,
    cumulative_usd,
    detect_plateau,
    fields_corrected_share,
    load_rows,
    median_review_seconds,
    suspect_count,
    upsert_row,
)
from vex.schema import FIELD_NAMES
from tests.conftest import VALID_VALUES


def _rec(doc_id, status, seconds=None, **values):
    return AnnotationRecord(doc_id=doc_id, round=2, status=status, values={**VALID_VALUES, **values},
                            source="model" if status == "pre_annotated" else "human", annotator="local:t",
                            backend="local", review_seconds=seconds)


def test_corrected_share_and_median_review_time(tmp_path, config):
    store = AnnotationStore(tmp_path)
    store.append([_rec("a", "pre_annotated", immatriculation="X"), _rec("a", "reviewed", seconds=60),
                  _rec("b", "pre_annotated"), _rec("b", "reviewed", seconds=120, assureur="Autre"),
                  _rec("c", "pre_annotated")])  # c not reviewed: ignored
    assert fields_corrected_share(store, 2, config.metric) == pytest.approx(2 / (2 * len(FIELD_NAMES)))
    assert median_review_seconds(store, 2) == 90
    assert fields_corrected_share(store, 3, config.metric) is None and median_review_seconds(store, 3) is None


def test_suspects_and_cumulative_usd(tmp_path):
    (tmp_path / "q").mkdir()
    (tmp_path / "q" / "suspect_annotations.jsonl").write_text('{"a":1}\n{"b":2}\n')
    assert suspect_count(tmp_path / "q") == 2 and suspect_count(tmp_path / "missing") == 0
    for name, usd in (("r1", 0.5), ("r2", 0.25)):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "cost.json").write_text(json.dumps({"total_usd": usd}))
    assert cumulative_usd(tmp_path / "runs") == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([0.80], False), ([0.80, 0.805, 0.807], True), ([0.80, 0.85, 0.855], False), ([0.80, 0.805, 0.79], True)],
)
def test_detect_plateau(scores, expected):
    assert detect_plateau(scores, delta=0.01, rounds=2) is expected


def _heldout(n_blind: int, standin_score: float | None):
    report = {"global_score": standin_score, "field_scores": dict.fromkeys(FIELD_NAMES, standin_score)}
    return {"heldout_blind": {"n": n_blind, "scores": {}},
            "heldout_standin": {"n": 15, "scores": {"current": report} if standin_score is not None else {}}}


def test_rows_upsert_by_round_and_flag_plateau(tmp_path):
    csv_path = tmp_path / "rounds.csv"
    rows = []
    for round_number, score, usd in ((1, 0.80, 0.10), (2, 0.805, 0.30), (3, 0.807, 0.40)):
        row = build_row(round_number, "abcdef123456", _heldout(0, score), 10, 6, 0.1, 75.0, 0, usd, rows, 0.01, 2)
        rows = upsert_row(csv_path, row)
    assert [r["round"] for r in rows] == ["1", "2", "3"]
    assert list(rows[0]) == COLUMNS
    assert rows[-1]["heldout_split"] == "heldout_standin" and rows[-1]["plateau"] == "True"
    assert rows[1]["usd_round"] == "0.2000"
    again = build_row(2, "abcdef123456", _heldout(0, 0.9), 10, 6, None, None, 1, 0.3, rows[:1], 0.01, 2)
    rows = upsert_row(csv_path, again)
    assert len(load_rows(csv_path)) == 3 and rows[1]["heldout_global"] == "0.9000"
    assert rows[1]["fields_corrected_share"] == "" and rows[1]["median_review_seconds"] == ""


def test_row_without_heldout_gold_leaves_score_empty():
    row = build_row(1, "abc", _heldout(0, None), 0, 0, None, None, 0, 0.0, [], 0.01, 2)
    assert row["heldout_global"] == "" and row["plateau"] == "False"

"""Per-field comparison, error typing, document scores, reports, sanity runs."""

import json

import pydantic
import pytest

from vex.config import MetricConfig, load_config
from vex.data.registry import DocRecord
from vex.data.sanity import eligible_documents, run_sanity
from vex.data.splits import Splits
from vex.metric.compare import compare_field, score_document, set_f1
from vex.metric.report import build_report
from vex.prompt import seed_candidate
from vex.schema import FIELD_NAMES
from tests.conftest import DEFAULT_YAML, VALID_VALUES, chat_response, make_pdf

METRIC = load_config(DEFAULT_YAML).metric


@pytest.mark.parametrize(
    ("field", "gold", "pred", "error", "score"),
    [
        ("immatriculation", "AB-123-CD", "ab 123 cd", "correct", 1.0),
        ("immatriculation", "AB-123-CD", "AB-123-CO", "wrong_value", 0.0),
        ("immatriculation", "AB-123-CD", None, "omission", 0.0),
        ("immatriculation", None, "AB-123-CD", "hallucination", 0.0),
        ("immatriculation", None, None, "correct", 1.0),
        ("immatriculation", "", None, "correct", 1.0),
        ("date_debut_validite", "01/02/2025", "2025-02-01", "correct", 1.0),
        ("date_debut_validite", "2025-02-01", "2025-01-02", "wrong_value", 0.0),
        ("numero_contrat", "AF-77-31-552", "AF7731552", "correct", 1.0),
        ("code_assureur", "0457", "457", "wrong_value", 0.0),
        ("marque_vehicule", "Citroën", "CITROEN", "correct", 1.0),
        ("marque_vehicule", "Mercedes-Benz", "Mercedes", "wrong_value", 0.0),  # not a fuzzy field
        ("nom_souscripteur", "Jean-Baptiste Leroux", "Jean Baptiste Leroux", "correct", 1.0),
        ("nom_souscripteur", "Leroux Jean-Baptiste", "Jean-Baptiste Leroux", "correct", 1.0),  # token sort
        ("nom_souscripteur", "Jean-Baptiste Leroux", "Jean-Baptiste Lerous", "correct", 1.0),  # 1 typo
        ("nom_souscripteur", "Jean-Baptiste Leroux", "Camille Martin", "wrong_value", 0.0),
        ("adresse_souscripteur", "12 rue des Lilas, 69003 Lyon", "12, rue des Lilas 69003 LYON", "correct", 1.0),
        ("adresse_souscripteur", "rue de Julien, 65272 Pottiernec", "rue de Julien, 65272 Pottier nec", "correct", 1.0),
        ("adresse_souscripteur", "rue de Julien, 65272 Pottiernec", "8 avenue Foch, 75016 Paris", "wrong_value", 0.0),
        ("garanties", ["Vol", "Incendie"], ["incendie", "VOL"], "correct", 1.0),
        ("garanties", ["Vol", "Incendie"], ["Vol"], "wrong_value", pytest.approx(2 / 3)),
        ("garanties", ["Vol"], [], "omission", 0.0),
        ("garanties", None, ["Vol"], "hallucination", 0.0),
    ],
)
def test_compare_field(field, gold, pred, error, score):
    result = compare_field(field, gold, pred, METRIC)
    assert result.error == error
    assert result.score == score


def test_fuzzy_threshold_is_configurable():
    strict = METRIC.model_copy(update={"fuzzy_threshold": 100.0})
    assert compare_field("nom_souscripteur", "Jean-Baptiste Leroux", "Jean-Baptiste Lerous", strict).error == "wrong_value"


def test_diff_line_says_what_was_expected_and_got():
    diff = compare_field("immatriculation", "AB-123-CD", "AB-123-CO", METRIC).diff
    assert diff.startswith("[wrong_value")
    assert "attendu/expected 'AB-123-CD'" in diff and "obtenu/got 'AB-123-CO'" in diff
    garanties = compare_field("garanties", ["Vol", "Incendie"], ["Vol", "Assistance"], METRIC).diff
    assert "missing ['incendie']" in garanties and "extra ['assistance']" in garanties
    assert "omission" in compare_field("assureur", "X", None, METRIC).diff


def test_set_f1():
    assert set_f1(frozenset("ab"), frozenset("ab")) == 1.0
    assert set_f1(frozenset("ab"), frozenset("cd")) == 0.0


def test_document_score_is_weighted_mean():
    pred = {**VALID_VALUES, "immatriculation": "ZZ-999-ZZ"}  # weight 3 of 14
    doc = score_document("d1", VALID_VALUES, pred, METRIC)
    assert doc.score == pytest.approx(1 - 3 / sum(METRIC.weights.values()))
    assert not doc.parse_failed
    assert doc.feedback().count("\n") == 0 and "immatriculation" in doc.feedback()
    assert score_document("d2", VALID_VALUES, VALID_VALUES, METRIC).feedback() == ""


def test_parse_failure_types_every_field_as_omission():
    gold = {**VALID_VALUES, "code_assureur": None}
    doc = score_document("d", gold, None, METRIC)
    assert doc.parse_failed and doc.score == 0.0
    assert [r.error for r in doc.fields] == ["omission"] * len(FIELD_NAMES)


def test_metric_config_must_cover_schema():
    data = METRIC.model_dump()
    with pytest.raises(pydantic.ValidationError):
        MetricConfig.model_validate({**data, "weights": {**data["weights"], "vin": 1.0}})
    with pytest.raises(pydantic.ValidationError):
        MetricConfig.model_validate({**data, "fuzzy_fields": ["vin"]})


def test_report_aggregates_fields_and_errors():
    perfect = score_document("a", VALID_VALUES, VALID_VALUES, METRIC)
    wrong = score_document("b", VALID_VALUES, {**VALID_VALUES, "assureur": None, "code_assureur": "9"}, METRIC)
    failed = score_document("c", VALID_VALUES, None, METRIC)
    report = build_report([perfect, wrong, failed])
    assert report.n_documents == 3 and report.parse_failures == 1
    assert report.global_score == pytest.approx((perfect.score + wrong.score + 0.0) / 3)
    assert report.field_scores["assureur"] == pytest.approx(1 / 3)
    assert report.errors["assureur"] == {"correct": 1, "omission": 2, "hallucination": 0, "wrong_value": 0}
    assert report.errors["code_assureur"]["wrong_value"] == 1
    assert "GLOBAL" in report.table() and json.dumps(report.to_dict())
    assert build_report([]).global_score == 0.0


def _record(tmp_path, name: str, origin: str = "synthetic") -> DocRecord:
    path = make_pdf(tmp_path / name)
    return DocRecord(doc_id=name, origin=origin, path=str(path), pages=1, added_at="t")


def test_sanity_only_uses_gepa_buckets(tmp_path):
    docs = [_record(tmp_path, f"d{i}.pdf") for i in range(4)] + [_record(tmp_path, "r.pdf", origin="real")]
    splits = Splits(seed=0, heldout_standin=["d0.pdf"], reserved_unused=["d1.pdf"],
                    gepa_train=["d2.pdf", "r.pdf"], gepa_pareto=["d3.pdf"])
    assert [d.doc_id for d in eligible_documents(docs, splits)] == ["d2.pdf", "d3.pdf"]


async def test_run_sanity_scores_against_meta(tmp_path, make_client, fake_openrouter):
    docs = [_record(tmp_path, "syn_0000_memo.pdf"), _record(tmp_path, "syn_0001_memo.pdf")]
    meta_path = tmp_path / "meta.jsonl"
    meta_path.write_text("\n".join(json.dumps({"file": d.doc_id, "values": VALID_VALUES}) for d in docs))
    fake_openrouter.queue += [chat_response(json.dumps(VALID_VALUES)), chat_response(json.dumps(VALID_VALUES))]
    async with make_client() as client:
        scores = await run_sanity(client, seed_candidate(), docs, meta_path, n=5, seed=0)
    assert len(scores) == 2 and all(s.score == 1.0 for s in scores)

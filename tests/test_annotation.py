"""Annotation store, local backend, Kili payloads (offline) and Kili backend (mocked SDK)."""

import json
from pathlib import Path

import pydantic
import pytest

import kili.client  # noqa: F401  (2.142.1: must be imported before the label parser)
from kili.services.label_data_parsing.json_response import ParsedJobs
from kili.services.label_data_parsing.types import Project

from vex.annotation.backend import PushItem, clean_human_values
from vex.annotation.kili_backend import KiliBackend, MissingKiliCredentialsError, as_pdf
from vex.annotation.kili_payloads import (
    COMMENT_JOB,
    asset_metadata,
    json_interface,
    label_to_record,
    latest_labels,
    prediction_response,
)
from vex.annotation.local import LocalBackend
from vex.annotation.store import AnnotationRecord, AnnotationStore, effective_split, held_out_ids
from vex.config import Settings
from vex.data.splits import Splits
from vex.schema import FIELD_NAMES
from tests.conftest import VALID_VALUES, make_pdf


def _record(doc_id: str, status: str, round_number: int = 1, created_at: str = "2026-10-17T10:00:00+00:00",
            **values) -> AnnotationRecord:
    return AnnotationRecord(doc_id=doc_id, round=round_number, status=status, values={**VALID_VALUES, **values},
                            source="model" if status == "pre_annotated" else "human", annotator="local:t",
                            backend="local", created_at=created_at)


# --- store -------------------------------------------------------------------


def test_only_reviewed_and_blind_are_gold(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append([_record("a", "pre_annotated"), _record("b", "reviewed"), _record("c", "blind", round_number=0)])
    assert set(store.gold()) == {"b", "c"}


def test_latest_human_record_wins_and_pre_annotation_never_overrides(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append([
        _record("a", "reviewed", assureur="Old"),
        _record("a", "reviewed", created_at="2026-10-18T10:00:00+00:00", assureur="Corrected"),
        _record("a", "pre_annotated", round_number=3, assureur="Model"),
    ])
    assert store.gold()["a"].values["assureur"] == "Corrected"


def test_append_new_is_idempotent(tmp_path):
    store = AnnotationStore(tmp_path)
    record = _record("a", "reviewed")
    assert len(store.append_new([record])) == 1
    duplicate = record.model_copy(update={"record_id": "other"})
    assert store.append_new([duplicate]) == []
    assert len(store.records()) == 1


def test_records_validate_values_and_refuse_generator_labels():
    assert _record("a", "reviewed", assureur="  ").values["assureur"] is None
    with pytest.raises(pydantic.ValidationError):
        _record("a", "reviewed", date_debut_validite="01/02/2025")  # backends must normalize first
    with pytest.raises(pydantic.ValidationError):
        AnnotationRecord(doc_id="a", round=1, status="reviewed", values=VALID_VALUES, source="human",
                         annotator="synthetic_meta", backend="local")


def test_effective_split_is_bucket_intersect_gold_and_refuses_held_out(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append([_record("t1", "reviewed"), _record("t2", "pre_annotated"), _record("p1", "reviewed")])
    splits = Splits(seed=0, heldout_blind=["h"], heldout_standin=["s"], reserved_unused=["r"],
                    gepa_train=["t1", "t2", "t3"], gepa_pareto=["p1"])
    assert effective_split("gepa_train", splits, store) == ["t1"]
    assert effective_split("gepa_pareto", splits, store) == ["p1"]
    with pytest.raises(ValueError):
        effective_split("heldout_blind", splits, store)  # type: ignore[arg-type]
    assert held_out_ids(splits) == {"h", "s", "r"}


def test_clean_human_values_normalizes_dates_and_guarantee_text():
    values = clean_human_values({**VALID_VALUES, "date_debut_validite": "1er février 2025",
                                 "date_fin_validite": "31/01/2026", "garanties": "Vol; Incendie\nBris de glace"})
    assert values["date_debut_validite"] == "2025-02-01" and values["date_fin_validite"] == "2026-01-31"
    assert values["garanties"] == ["Vol", "Incendie", "Bris de glace"]
    with pytest.raises(pydantic.ValidationError):
        clean_human_values({**VALID_VALUES, "date_debut_validite": "début mars"})


def test_blind_push_item_cannot_carry_values(tmp_path):
    with pytest.raises(ValueError):
        PushItem("d", tmp_path / "x.pdf", "real", "blind", VALID_VALUES)


# --- local backend -------------------------------------------------------------


def test_local_round_trip(tmp_path):
    backend = LocalBackend(tmp_path / "inbox")
    backend.push([
        PushItem("doc_pre_0000000000", tmp_path / "a.pdf", "synthetic", "pre_annotated", VALID_VALUES),
        PushItem("doc_blind_00000000", tmp_path / "b.pdf", "real", "blind", None),
    ], round_number=2)
    folder = tmp_path / "inbox" / "round_2"
    pre, blind = folder / "doc_pre_0000000000.json", folder / "doc_blind_00000000.json"
    assert json.loads(blind.read_text())["values"] == dict.fromkeys(FIELD_NAMES)
    assert backend.pull(2) == []  # nothing marked done yet

    edited = json.loads(pre.read_text())
    edited.update(_done=True, annotator="antoine", comment="deux véhicules", review_seconds=42)
    edited["values"]["immatriculation"] = "ab 123 cd"
    edited["values"]["date_debut_validite"] = "01/02/2025"
    pre.write_text(json.dumps(edited))
    blank = json.loads(blind.read_text())
    blank.update(_done=True, values={**dict.fromkeys(FIELD_NAMES), "date_fin_validite": "pas une date"})
    blind.write_text(json.dumps(blank))

    [record] = backend.pull(2)  # the blind file has an invalid date: reported, not imported
    assert record.status == "reviewed" and record.annotator == "local:antoine"
    assert record.values["date_debut_validite"] == "2025-02-01" and record.review_seconds == 42
    assert record.comment == "deux véhicules"
    assert (folder / "doc_pre_0000000000.imported.json").exists() and blind.exists()
    assert backend.pull(2) == []


# --- Kili payloads (validated with the SDK's own parser, offline) -------------


def _parse(response: dict) -> ParsedJobs:
    return ParsedJobs(project_info=Project(jsonInterface=json_interface()["jobs"], inputType="PDF"),
                      json_response=response)


def test_interface_has_one_optional_transcription_job_per_field_plus_comment():
    jobs = json_interface()["jobs"]
    assert set(jobs) == {name.upper() for name in FIELD_NAMES} | {COMMENT_JOB}
    assert all(job["mlTask"] == "TRANSCRIPTION" and job["required"] == 0 for job in jobs.values())
    assert "PREMIER" in jobs["IMMATRICULATION"]["instruction"]


def test_prediction_response_is_accepted_by_kili_parser_and_omits_nulls():
    response = prediction_response({**VALID_VALUES, "code_assureur": None})
    assert "CODE_ASSUREUR" not in response
    assert response["GARANTIES"]["text"] == "Responsabilité civile\nVol\nBris de glace"
    parsed = _parse(json.loads(json.dumps(response)))
    assert parsed["IMMATRICULATION"].text == "AB-123-CD"


def test_unknown_job_is_rejected_by_kili_parser():
    with pytest.raises(Exception):
        _parse({"VIN": {"text": "x"}})


def test_asset_metadata():
    assert asset_metadata("d", "real", "blind", 0) == {"doc_id": "d", "origin": "real", "status": "blind", "round": 0}


def _label(external_id: str, label_type: str = "DEFAULT", created_at: str = "2026-10-18T09:00:00Z", **texts) -> dict:
    response = {job: {"text": text} for job, text in texts.items()}
    return {"jsonResponse": response, "labelType": label_type, "createdAt": created_at, "secondsToLabel": 75,
            "author": {"email": "annot@corp.fr"}, "labelOf": {"externalId": external_id}}


def test_label_to_record_maps_empty_text_to_null_and_reads_review_time():
    label = _label("doc1", IMMATRICULATION="ab 123 cd", DATE_DEBUT_VALIDITE="01/02/2025", ASSUREUR="  ",
                   GARANTIES="Vol\nIncendie", COMMENTAIRE="flou en bas")
    record = label_to_record(label, "reviewed", 3)
    assert record.doc_id == "doc1" and record.round == 3 and record.backend == "kili"
    assert record.values["assureur"] is None and record.values["immatriculation"] == "ab 123 cd"
    assert record.values["date_debut_validite"] == "2025-02-01" and record.values["garanties"] == ["Vol", "Incendie"]
    assert record.review_seconds == 75.0 and record.comment == "flou en bas"
    assert record.annotator == "kili:annot@corp.fr"


def test_latest_labels_prefers_review_then_most_recent():
    labels = [_label("a", created_at="2026-01-02"), _label("a", "REVIEW", created_at="2026-01-01"),
              _label("b", created_at="2026-01-01"), _label("b", created_at="2026-01-03")]
    chosen = latest_labels(labels)
    assert chosen["a"]["labelType"] == "REVIEW" and chosen["b"]["createdAt"] == "2026-01-03"


# --- Kili backend (SDK mocked: no network) ------------------------------------


class FakeKili:
    """Records SDK calls; returns canned assets and labels."""

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_project(self, **kwargs):
        self.calls.append(("create_project", kwargs))
        return {"id": "proj-1"}

    def append_many_to_dataset(self, **kwargs):
        self.calls.append(("append_many_to_dataset", kwargs))
        assert all(Path(p).suffix == ".pdf" and Path(p).exists() for p in kwargs["content_array"])

    def create_predictions(self, **kwargs):
        self.calls.append(("create_predictions", kwargs))

    def assets(self, **kwargs):
        self.calls.append(("assets", kwargs))
        return [{"externalId": "pre", "jsonMetadata": asset_metadata("pre", "synthetic", "pre_annotated", 1)},
                {"externalId": "bl", "jsonMetadata": asset_metadata("bl", "real", "blind", 1)}]

    def labels(self, **kwargs):
        self.calls.append(("labels", kwargs))
        return [_label("pre", IMMATRICULATION="AB-123-CD"), _label("bl", ASSUREUR="X")]


@pytest.fixture
def kili_backend(config, monkeypatch):
    fake = FakeKili()
    monkeypatch.setattr(kili.client, "Kili", lambda **kwargs: fake)
    settings = Settings(_env_file=None, kili_api_key="k", kili_api_endpoint="https://kili.corp/api")
    return KiliBackend(config, settings), fake


def test_kili_backend_requires_credentials(config):
    with pytest.raises(MissingKiliCredentialsError):
        KiliBackend(config, Settings(_env_file=None))


def test_kili_push_uploads_pdfs_and_skips_predictions_for_blind(kili_backend, tmp_path):
    backend, fake = kili_backend
    png = tmp_path / "scan.png"
    from PIL import Image
    Image.new("RGB", (20, 30), "white").save(png)
    backend.push([PushItem("pre", make_pdf(tmp_path / "a.pdf"), "synthetic", "pre_annotated", VALID_VALUES),
                  PushItem("bl", png, "real", "blind", None)], round_number=1)
    names = [name for name, _ in fake.calls]
    assert names == ["create_project", "append_many_to_dataset", "create_predictions"]
    upload = fake.calls[1][1]
    assert upload["external_id_array"] == ["pre", "bl"]
    assert upload["json_metadata_array"][1]["status"] == "blind"
    predictions = fake.calls[2][1]
    assert predictions["external_id_array"] == ["pre"] and predictions["model_name"] == "vex-round1"


def test_kili_pull_converts_labels_with_status_from_metadata(kili_backend):
    backend, fake = kili_backend
    records = {r.doc_id: r for r in backend.pull(1)}
    assert records["pre"].status == "reviewed" and records["bl"].status == "blind"
    labels_call = dict(fake.calls)["labels"]
    assert labels_call["type_in"] == ["DEFAULT", "REVIEW"] and "labelOf.externalId" in labels_call["fields"]


def test_as_pdf_wraps_images(tmp_path):
    from PIL import Image
    png = tmp_path / "x.png"
    Image.new("RGB", (40, 60), "white").save(png)
    assert as_pdf(png, tmp_path).suffix == ".pdf"
    pdf = make_pdf(tmp_path / "y.pdf")
    assert as_pdf(pdf, tmp_path) == pdf

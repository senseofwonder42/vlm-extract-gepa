"""Bootstrap rounds: selection, corrections, rewrite proposal, end-to-end CLI flow."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vex import cli
from vex.annotation.store import AnnotationRecord, AnnotationStore
from vex.bootstrap import (
    FieldRewrite,
    apply_rewrites,
    candidate_diff,
    collect_corrections,
    propose_rewrites,
    select_batch,
)
from vex.data.registry import DocRecord
from vex.data.splits import Splits
from vex.prompt import load_candidate, seed_candidate
from tests.conftest import DEFAULT_YAML, VALID_VALUES, chat_response, make_pdf


def _doc(doc_id: str, origin: str = "synthetic", path: str = "x.pdf") -> DocRecord:
    return DocRecord(doc_id=doc_id, origin=origin, path=path, pages=1, added_at="t")


SPLITS = Splits(seed=0, heldout_blind=["h1"], heldout_standin=["s1"], reserved_unused=["r1"],
                gepa_train=["t1", "t2", "t3", "real1"], gepa_pareto=["p1", "p2"])
REGISTRY = [_doc(d) for d in ("h1", "s1", "r1", "t1", "t2", "t3", "p1", "p2")] + [_doc("real1", origin="real")]


def _store_record(doc_id: str, status: str, round_number: int = 1, comment: str | None = None, **values):
    return AnnotationRecord(doc_id=doc_id, round=round_number, status=status, values={**VALID_VALUES, **values},
                            source="model" if status == "pre_annotated" else "human", annotator="local:t",
                            backend="local", comment=comment)


def test_select_batch_real_first_skips_held_out_and_annotated(tmp_path):
    store = AnnotationStore(tmp_path)
    store.append([_store_record("t1", "reviewed")])
    batch = select_batch(1, 4, REGISTRY, SPLITS, store, seed=0)
    ids = [d.doc_id for d in batch]
    assert ids[0] == "real1" and len(ids) == 4
    assert not set(ids) & {"h1", "s1", "r1", "t1"}
    assert ids == [d.doc_id for d in select_batch(1, 4, REGISTRY, SPLITS, store, seed=0)]


def test_collect_corrections_only_changed_fields(tmp_path, config):
    store = AnnotationStore(tmp_path)
    store.append([
        _store_record("t1", "pre_annotated", immatriculation="AB-123-CO", garanties=["Vol", "Assistance 24H/24"]),
        _store_record("t1", "reviewed", comment="le O est un D", immatriculation="ab 123 cd",
                      garanties=["Vol"]),
        _store_record("t2", "pre_annotated"),  # not reviewed yet: ignored
    ])
    corrections = collect_corrections(store, 1, SPLITS, config.metric)
    assert sorted(c.field for c in corrections) == ["garanties", "immatriculation"]
    assert all(c.comment == "le O est un D" for c in corrections)


def test_collect_corrections_refuses_held_out_documents(tmp_path, config):
    store = AnnotationStore(tmp_path)
    store.append([_store_record("h1", "pre_annotated")])
    with pytest.raises(RuntimeError, match="Held-out"):
        collect_corrections(store, 1, SPLITS, config.metric)


async def test_propose_rewrites_sends_no_doc_ids(make_client, fake_openrouter, tmp_path, config):
    store = AnnotationStore(tmp_path)
    store.append([_store_record("t1", "pre_annotated", immatriculation="X"),
                  _store_record("t1", "reviewed", immatriculation="AB-123-CD")])
    proposal = {"rewrites": [{"field": "immatriculation", "description": "Nouvelle règle. Si absent → null.",
                              "rationale": "confusion"}]}
    fake_openrouter.queue.append(chat_response(json.dumps(proposal)))
    async with make_client() as client:
        rewrites = await propose_rewrites(client, seed_candidate(), collect_corrections(store, 1, SPLITS, config.metric))
    assert rewrites[0].field == "immatriculation"
    sent = json.dumps(fake_openrouter.chat_requests[0]["messages"])
    assert "t1" not in sent  # the rewriter sees values and comments, never document ids
    assert fake_openrouter.chat_requests[0]["model"] == config.role("reflection").model


def test_apply_and_diff():
    old = seed_candidate()
    new = apply_rewrites(old, [FieldRewrite(field="assureur", description="Texte. Nouveau. Si absent → null.",
                                            rationale="r")])
    assert new["field:assureur"].startswith("Texte") and old["field:assureur"] != new["field:assureur"]
    diff = candidate_diff(old, new)
    assert "field:assureur (proposed)" in diff and "field:immatriculation" not in diff


def test_bootstrap_cli_end_to_end_with_local_backend(tmp_path, monkeypatch, fake_openrouter):
    """push → human edits inbox → pull + rewrite proposal → saved candidate."""
    raw = tmp_path / "raw"
    docs = [make_pdf(tmp_path / f"d{i}.pdf") for i in range(3)]
    registry = tmp_path / "registry.jsonl"
    registry.write_text("\n".join(_doc(f"doc{i:062d}", path=str(p)).model_dump_json() for i, p in enumerate(docs)))
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text(json.dumps({"seed": 42, "gepa_train": [f"doc{i:062d}" for i in range(3)]}))
    original = cli.OpenRouterClient
    monkeypatch.setattr(cli, "OpenRouterClient",
                        lambda *a, **k: original(*a, **k, transport=fake_openrouter.transport()))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-0000000000000000")
    args = ["--config", str(DEFAULT_YAML)] + [
        x for key, value in {
            "paths.raw": raw, "paths.registry": registry, "paths.splits": splits_path,
            "paths.annotations": tmp_path / "ann", "paths.inbox": tmp_path / "inbox",
            "paths.prompts": tmp_path / "prompts", "paths.runs": tmp_path / "runs", "cache.dir": tmp_path / "cache",
            "bootstrap.batch_sizes": "[2, 3]",
        }.items() for x in ("--set", f"{key}={value}")
    ]
    runner = CliRunner()

    wrong = {**VALID_VALUES, "garanties": ["Vol", "Assistance 24H/24"]}
    fake_openrouter.queue += [chat_response(json.dumps(wrong)), chat_response(json.dumps(wrong))]
    first = runner.invoke(cli.app, [*args, "bootstrap", "--round", "1"])
    assert first.exit_code == 0, first.output
    inbox = sorted((tmp_path / "inbox" / "round_1").glob("*.json"))
    assert len(inbox) == 2

    second = runner.invoke(cli.app, [*args, "bootstrap", "--round", "1"])
    assert "0/2 done" in second.output

    for path in inbox:
        payload = json.loads(path.read_text())
        payload["values"]["garanties"] = ["Vol"]
        payload.update(_done=True, comment="Assistance 24H/24 est un numéro, pas une garantie")
        path.write_text(json.dumps(payload))
    proposal = {"rewrites": [{"field": "garanties", "description": "Ne pas lister les numéros d'assistance. "
                              "Si aucune garantie → null.", "rationale": "assistance phone read as guarantee"}]}
    fake_openrouter.queue.append(chat_response(json.dumps(proposal)))
    third = runner.invoke(cli.app, [*args, "bootstrap", "--round", "1", "--yes"])
    assert third.exit_code == 0, third.output
    assert "field:garanties (proposed)" in third.output
    [saved] = (tmp_path / "prompts").glob("bootstrap-r1.*.json")
    assert "numéros d'assistance" in load_candidate(saved)["field:garanties"]
    store = AnnotationStore(tmp_path / "ann")
    assert {r.status for r in store.records()} == {"pre_annotated", "reviewed"}
    assert len(store.gold()) == 2

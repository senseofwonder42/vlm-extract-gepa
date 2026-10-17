"""Active learning: signals, combination, diversity, batch planning, select end-to-end."""

import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest

from vex.active.diversity import _preprocess, embed_documents, pick_diverse, thumbnail_embeddings
from vex.active.select import DocSignals, build_pool, compute_signals, current_candidates, plan_batch
from vex.active.signals import (
    combine,
    disagreement,
    pareto_disagreement,
    rule_violations,
    rules_score,
    self_consistency,
    verifier_score,
)
from vex.annotation.store import AnnotationRecord, AnnotationStore
from vex.data.registry import DocRecord
from vex.data.splits import Splits
from vex.llm.client import OpenRouterClient
from vex.llm.cost import CostTracker
from vex.llm.media import load_pages
from vex.prompt import save_candidate, seed_candidate
from tests.conftest import FAKE_KEY, VALID_VALUES, chat_response, endpoints_response, make_pdf

TODAY = date(2026, 10, 17)


@pytest.fixture
def weights(config):
    return config.metric.weights


# --- signals -----------------------------------------------------------------


def test_disagreement_is_zero_when_unanimous_after_normalization(weights):
    a = VALID_VALUES
    b = {**VALID_VALUES, "immatriculation": "ab 123 cd", "date_debut_validite": "2025-02-01"}
    assert disagreement([a, b, a], weights) == 0.0
    assert disagreement([a], weights) == 0.0


def test_disagreement_is_weighted(weights):
    plate = {**VALID_VALUES, "immatriculation": "ZZ-999-ZZ"}   # weight 3
    brand = {**VALID_VALUES, "marque_vehicule": "Renault"}      # weight 0.5
    assert disagreement([VALID_VALUES, plate], weights) > disagreement([VALID_VALUES, brand], weights) > 0


def test_unparseable_sample_disagrees_everywhere(weights):
    assert disagreement([VALID_VALUES, None], weights) == pytest.approx(0.5)
    assert pareto_disagreement([None, None], weights) == pytest.approx(0.5)


def test_self_consistency_mixes_samples_and_haiku(weights):
    assert self_consistency([VALID_VALUES] * 3, VALID_VALUES, VALID_VALUES, weights) == 0.0
    other = {**VALID_VALUES, "assureur": "Autre"}
    only_haiku = self_consistency([VALID_VALUES] * 3, VALID_VALUES, other, weights)
    assert 0 < only_haiku == pytest.approx(disagreement([VALID_VALUES, other], weights) / 2)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, []),
        ({"assureur": None}, ["missing:assureur"]),
        ({"date_fin_validite": "2025-01-01"}, ["end_before_start"]),
        ({"date_debut_validite": "2015-01-01"}, ["implausible_date:date_debut_validite"]),
        ({"immatriculation": "VF1RFB00X12345678"}, ["bad_plate"]),
        ({"immatriculation": "1234 AB 56"}, []),
    ],
)
def test_rule_violations(changes, expected):
    assert rule_violations({**VALID_VALUES, **changes}, TODAY) == expected


def test_rules_score_bounds():
    assert rules_score(VALID_VALUES, TODAY) == 0.0
    assert rules_score(None, TODAY) == 1.0
    assert 0 < rules_score({**VALID_VALUES, "assureur": None}, TODAY) < 1


def test_combine_ignores_missing_signals():
    w = {"pareto_disagreement": 0.35, "self_consistency": 0.25, "rules": 0.2, "verifier": 0.2}
    assert combine({"pareto_disagreement": None, "self_consistency": 1.0, "rules": 0.0, "verifier": None}, w) == \
        pytest.approx(0.25 / 0.45)
    assert combine({"pareto_disagreement": None, "self_consistency": None, "rules": None, "verifier": None}, w) == 0.0


async def test_verifier_score_counts_unquoted_values(make_client, fake_openrouter, pdf_path, weights):
    checks = [{"field": f, "found": True, "quote": "x"} for f in VALID_VALUES]
    checks[4] = {"field": "immatriculation", "found": False, "quote": ""}  # plate not found
    fake_openrouter.queue.append(chat_response(json.dumps({"checks": checks})))
    async with make_client() as client:
        score = await verifier_score(client, pdf_path, VALID_VALUES, weights)
        assert await verifier_score(client, pdf_path, None, weights) == 1.0
    assert score == pytest.approx(3 / sum(weights.values()))
    assert fake_openrouter.chat_requests[0]["model"] == "anthropic/claude-haiku-5.5"


# --- diversity ------------------------------------------------------------------


def test_thumbnail_embeddings_are_normalized(tmp_path):
    paths = [make_pdf(tmp_path / f"{i}.pdf", pages=1) for i in range(2)]
    vectors = thumbnail_embeddings(paths)
    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_documents_falls_back_without_onnx_extra(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_onnx(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_onnx)
    _, used = embed_documents([make_pdf(tmp_path / "a.pdf")], "auto")
    assert used == "thumbnail"
    with pytest.raises(ImportError):
        embed_documents([make_pdf(tmp_path / "b.pdf")], "onnx")


def test_onnx_preprocessing_shape_and_scale():
    from PIL import Image

    tensor = _preprocess(Image.new("RGB", (600, 900), "white"))
    assert tensor.shape == (3, 224, 224)
    assert tensor[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-4)


def test_pick_diverse_rotates_over_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(3)
    embeddings = np.vstack([centers[i] + rng.normal(0, 0.01, 3) for i in (0, 0, 0, 1, 1, 2)])
    ids = ["a1", "a2", "a3", "b1", "b2", "c1"]
    scores = {"a1": 0.9, "a2": 0.8, "a3": 0.7, "b1": 0.5, "b2": 0.4, "c1": 0.1}
    picked = pick_diverse(ids, embeddings, scores, n_clusters=3, k=3, seed=0)
    assert [doc for doc, _ in picked] == ["a1", "b1", "c1"]  # one per cluster before any second pick
    assert len({cluster for _, cluster in picked}) == 3
    assert len(pick_diverse(ids, embeddings, scores, n_clusters=3, k=10, seed=0)) == 6


# --- batch planning --------------------------------------------------------------


def _signals(tmp_path, n: int) -> list[DocSignals]:
    return [DocSignals(doc_id=f"d{i:02d}", path=str(make_pdf(tmp_path / f"d{i}.pdf")), origin="synthetic",
                       pareto=None, self_consistency=i / n, rules=0.0, verifier=None, combined=i / n,
                       haiku_values=VALID_VALUES, reference_values=VALID_VALUES) for i in range(n)]


def test_plan_batch_mixes_uncertain_and_random_confident(tmp_path, config):
    cfg = config.model_copy(update={"active": config.active.model_copy(update={
        "batch_size": 10, "random_confident_fraction": 0.2,
        "diversity": config.active.diversity.model_copy(update={"embedder": "thumbnail"})})})
    signals = _signals(tmp_path, 20)
    selections = plan_batch(signals, cfg, seed=1)
    assert len(selections) == 10 and len({s.doc_id for s in selections}) == 10
    random_picks = [s for s in selections if s.reason == "random_confident"]
    assert len(random_picks) == 2
    confident_half = {s.doc_id for s in sorted(signals, key=lambda s: -s.combined)[10:]}
    assert {s.doc_id for s in random_picks} <= confident_half
    assert all(s.doc_id not in confident_half for s in selections if s.reason == "uncertain")
    assert selections == plan_batch(signals, cfg, seed=1)


def test_build_pool_excludes_held_out_and_annotated(tmp_path):
    registry = [DocRecord(doc_id=d, origin="synthetic", path="x", pages=1, added_at="t")
                for d in ("t1", "t2", "p1", "h1", "s1", "r1")]
    splits = Splits(seed=0, heldout_blind=["h1"], heldout_standin=["s1"], reserved_unused=["r1"],
                    gepa_train=["t1", "t2"], gepa_pareto=["p1"])
    store = AnnotationStore(tmp_path)
    store.append([AnnotationRecord(doc_id="t1", round=1, status="pre_annotated", values=VALID_VALUES,
                                   source="model", annotator="model:x", backend="local")])
    assert sorted(r.doc_id for r in build_pool(registry, splits, store, 60, 0)) == ["p1", "t2"]
    assert len(build_pool(registry, splits, store, 1, 0)) == 1


def test_current_candidates_prefers_latest_optimize_run(tmp_path, config):
    cfg = config.model_copy(update={"paths": config.paths.model_copy(update={"runs": tmp_path / "runs",
                                                                              "prompts": tmp_path / "prompts"})})
    assert current_candidates(cfg, 3) == [seed_candidate()]
    other = {**seed_candidate(), "instructions": "autre"}
    save_candidate(other, cfg.paths.prompts, "bootstrap-r1")
    assert current_candidates(cfg, 3) == [other]
    run = tmp_path / "runs" / "20261017T000000Z-optimize-abcd1234"
    run.mkdir(parents=True)
    from vex.prompt import candidate_hash
    lines = [{"average_score": 0.5, "candidate": seed_candidate(), "candidate_hash": candidate_hash(seed_candidate())},
             {"average_score": 0.9, "candidate": other, "candidate_hash": candidate_hash(other)}]
    (run / "candidates.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    assert current_candidates(cfg, 3) == [other, seed_candidate()]


# --- compute_signals with a fake model -------------------------------------------


async def test_compute_signals_end_to_end(tmp_path, config):
    """A document where samples disagree must come out more uncertain than a stable one."""
    calls = {"n": 0}
    noisy_url: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints_response(True))
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        if name == "verification":
            checks = [{"field": f, "found": True, "quote": "q"} for f in VALID_VALUES]
            return httpx.Response(200, json=chat_response(json.dumps({"checks": checks})))
        is_noisy = noisy_url[0] in json.dumps(body["messages"][1]["content"])
        unstable = is_noisy and body.get("temperature", 0) > 0 and calls["n"] % 2 == 0  # samples alternate
        calls["n"] += is_noisy and body.get("temperature", 0) > 0
        values = {**VALID_VALUES, "immatriculation": "ZZ-999-ZZ"} if unstable else VALID_VALUES
        return httpx.Response(200, json=chat_response(json.dumps(values)))

    import pymupdf

    def doc(name: str, mark: bool) -> DocRecord:
        pdf = pymupdf.open()
        pdf.new_page().insert_text((50, 72), f"{name} {'MARK' if mark else ''}")
        path = tmp_path / f"{name}.pdf"
        pdf.save(path)
        return DocRecord(doc_id=name, origin="synthetic", path=str(path), pages=1, added_at="t")

    pool = [doc("stable", False), doc("noisy", True)]
    noisy_url.append(load_pages(Path(pool[1].path), config.render.dpi, 1)[0].content_part()["image_url"]["url"])
    async with OpenRouterClient(config, FAKE_KEY, CostTracker(5), use_cache=False,
                                transport=httpx.MockTransport(handler)) as client:
        signals = await compute_signals(client, pool, [seed_candidate()], config, TODAY)
    by_id = {s.doc_id: s for s in signals}
    assert by_id["stable"].pareto is None  # one candidate only
    assert by_id["stable"].verifier == 0.0 and by_id["stable"].haiku_values == VALID_VALUES
    assert by_id["noisy"].self_consistency > by_id["stable"].self_consistency

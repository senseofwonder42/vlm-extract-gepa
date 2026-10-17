"""GEPA adapter, error-driven selector, diagnostics, and an end-to-end gepa.optimize on a fake model."""

import json
from pathlib import Path

import httpx
import pymupdf
import pytest
import yaml

from vex.annotation.store import AnnotationRecord, AnnotationStore
from vex.config import Settings, load_config
from vex.diagnostic import Diagnostician
from vex.gepa_adapter import DocInstance, ErrorDrivenSelector, VexAdapter
from vex.llm.client import OpenRouterClient
from vex.llm.cost import BudgetExceeded, CostTracker
from vex.llm.media import load_pages
from vex.llm.sync import BackgroundLoop
from vex.optimize import run_optimize
from vex.prompt import load_candidate, seed_candidate
from tests.conftest import DEFAULT_YAML, FAKE_KEY, VALID_VALUES, chat_response, endpoints_response

FIX = "ne jamais lister le numéro d'assistance"
GOLD = {**VALID_VALUES, "garanties": None}  # e.g. a Mémo that lists no guarantee
HALLUCINATED = {**VALID_VALUES, "garanties": ["Assistance 24H/24"]}
DIAGNOSIS = {"located": True, "quote": "ASSISTANCE 24H/24 09 29 60 71 86", "location": "bloc bas du mémo",
             "explanation": "le numéro d'assistance a été pris pour une garantie"}


def unique_pdf(path: Path, label: str) -> Path:
    """A one-page PDF whose rendering differs per label (distinct image hashes)."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 72), f"MEMO VEHICULE ASSURE — {label}")
    page.insert_text((50, 100), "ASSISTANCE 24H/24 09 29 60 71 86")
    pdf.save(path)
    pdf.close()
    return path


class FakeModels:
    """Fake OpenRouter: extractor, diagnostic and reflection behaviours; records every request."""

    def __init__(self, diagnosis: dict = DIAGNOSIS) -> None:
        self.requests: list[dict] = []
        self.diagnosis = diagnosis

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints_response(True))
        body = json.loads(request.content)
        self.requests.append(body)
        name = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        if name == "attestation_auto":
            system = body["messages"][0]["content"]
            return httpx.Response(200, json=chat_response(json.dumps(GOLD if FIX in system else HALLUCINATED)))
        if name == "diagnosis":
            return httpx.Response(200, json=chat_response(json.dumps(self.diagnosis)))
        # GEPA's default reflection prompt; the new text must come back in a ``` block.
        return httpx.Response(200, json=chat_response(
            f"```\nListe des garanties souscrites ; {FIX} (c'est un service, pas une garantie). "
            "Si aucune garantie → null.\n```"))


def _client(config, fake: FakeModels, max_usd: float = 5.0) -> OpenRouterClient:
    return OpenRouterClient(config, FAKE_KEY, CostTracker(max_usd), use_cache=False,
                            transport=httpx.MockTransport(fake.handler))


@pytest.fixture
def loop():
    background = BackgroundLoop()
    yield background
    background.close()


@pytest.fixture
def adapter_parts(config, tmp_path, loop):
    fake = FakeModels()
    client = _client(config, fake)
    diag = Diagnostician(client, tmp_path / "diag", tmp_path / "review_queue")
    docs = [DocInstance(f"d{i}", unique_pdf(tmp_path / f"d{i}.pdf", f"d{i}"), GOLD) for i in range(3)]
    return VexAdapter(config, client, loop, diag), fake, docs


# --- evaluate ----------------------------------------------------------------


def test_evaluate_contract(adapter_parts):
    adapter, _, docs = adapter_parts
    batch = adapter.evaluate(docs, seed_candidate(), capture_traces=False)
    assert len(batch.outputs) == len(batch.scores) == 3 and batch.trajectories is None
    assert all(score == pytest.approx(1 - 1 / 14) for score in batch.scores)  # garanties weight 1 of 14
    traced = adapter.evaluate(docs, seed_candidate(), capture_traces=True)
    [garanties] = [f for f in traced.trajectories[0]["fields"] if f["error"] != "correct"]
    assert garanties["field"] == "garanties" and garanties["error"] == "hallucination"


def test_evaluate_never_raises_for_one_failed_document(config, tmp_path, loop):
    def handler(request):
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints_response(True))
        return httpx.Response(400, json={"error": {"message": "bad image"}})

    client = OpenRouterClient(config, FAKE_KEY, CostTracker(5), use_cache=False, transport=httpx.MockTransport(handler))
    adapter = VexAdapter(config, client, loop, Diagnostician(client, tmp_path / "d", tmp_path / "q"))
    batch = adapter.evaluate([DocInstance("x", unique_pdf(tmp_path / "x.pdf", "x"), GOLD)], seed_candidate(), True)
    assert batch.scores == [0.0] and "bad image" in batch.trajectories[0]["error"]


def test_budget_exceeded_propagates(config, tmp_path, loop):
    fake = FakeModels()
    client = _client(config, fake, max_usd=0.0)
    adapter = VexAdapter(config, client, loop, Diagnostician(client, tmp_path / "d", tmp_path / "q"))
    with pytest.raises(BudgetExceeded):
        adapter.evaluate([DocInstance("x", unique_pdf(tmp_path / "x.pdf", "x"), GOLD)], seed_candidate())


# --- reflective dataset --------------------------------------------------------


def test_reflective_dataset_routes_to_field_and_instructions(adapter_parts):
    adapter, fake, docs = adapter_parts
    candidate = seed_candidate()
    batch = adapter.evaluate(docs, candidate, capture_traces=True)
    dataset = adapter.make_reflective_dataset(candidate, batch, ["field:garanties", "instructions"])
    assert set(dataset) == {"field:garanties", "instructions"}
    record = dataset["field:garanties"][0]
    assert set(record) == {"Inputs", "Generated Outputs", "Feedback"}
    assert "hallucination" in record["Feedback"] and "ASSISTANCE 24H/24" in record["Feedback"]
    assert len(dataset["instructions"]) == 3
    diag_calls = sum(1 for r in fake.requests if (r.get("response_format") or {}).get("json_schema", {}).get("name") == "diagnosis")
    assert diag_calls == 3
    adapter.make_reflective_dataset(candidate, batch, ["field:garanties"])  # cached: no new diagnostic call
    assert sum(1 for r in fake.requests if (r.get("response_format") or {}).get("json_schema", {}).get("name") == "diagnosis") == 3


def test_only_requested_components_get_diagnosed(adapter_parts):
    adapter, fake, docs = adapter_parts
    batch = adapter.evaluate(docs, seed_candidate(), capture_traces=True)
    dataset = adapter.make_reflective_dataset(seed_candidate(), batch, ["field:immatriculation"])
    assert "No error" in dataset["field:immatriculation"][0]["Feedback"]
    assert not any((r.get("response_format") or {}).get("json_schema", {}).get("name") == "diagnosis" for r in fake.requests)


@pytest.mark.parametrize("diagnosis", [{**DIAGNOSIS, "located": False}, {**DIAGNOSIS, "quote": ""}])
def test_unlocated_and_unquoted_diagnoses_are_excluded(config, tmp_path, loop, diagnosis):
    fake = FakeModels(diagnosis)
    client = _client(config, fake)
    queue = tmp_path / "review_queue"
    adapter = VexAdapter(config, client, loop, Diagnostician(client, tmp_path / "diag", queue))
    docs = [DocInstance("d0", unique_pdf(tmp_path / "d0.pdf", "d0"), GOLD)]
    batch = adapter.evaluate(docs, seed_candidate(), capture_traces=True)
    dataset = adapter.make_reflective_dataset(seed_candidate(), batch, ["field:garanties"])
    assert "No error" in dataset["field:garanties"][0]["Feedback"]  # nothing usable was taught
    suspects = queue / "suspect_annotations.jsonl"
    assert suspects.exists() == (not diagnosis["located"])


# --- selector ------------------------------------------------------------------


def _traj(parsed=True, **field_scores):
    fields = [{"field": name, "score": field_scores.get(name, 1.0)} for name in VALID_VALUES]
    return {"parsed": {} if parsed else None, "fields": fields}


def test_error_driven_selector(config):
    select = ErrorDrivenSelector(config.metric.weights)
    assert select(None, [_traj(garanties=0.0), _traj(garanties=0.0)], [], 0, {}) == ["field:garanties"]
    # immatriculation weighs 3: one miss outweighs two garanties misses (weight 1)
    assert select(None, [_traj(garanties=0.0), _traj(garanties=0.0, immatriculation=0.0)], [], 0, {}) == [
        "field:immatriculation"]
    assert select(None, [_traj(parsed=False), _traj(garanties=0.0)], [], 0, {}) == ["instructions"]
    assert select(None, [_traj()], [], 0, {}) == ["instructions"]


# --- end-to-end gepa.optimize --------------------------------------------------


def test_run_optimize_end_to_end(tmp_path, monkeypatch):
    docs = {name: unique_pdf(tmp_path / f"{name}.pdf", name) for name in ("t1", "t2", "p1", "p2", "h1")}
    (tmp_path / "registry.jsonl").write_text("\n".join(
        json.dumps({"doc_id": name, "origin": "synthetic", "path": str(path), "pages": 1, "added_at": "t"})
        for name, path in docs.items()))
    (tmp_path / "splits.yaml").write_text(yaml.safe_dump(
        {"seed": 42, "gepa_train": ["t1", "t2"], "gepa_pareto": ["p1", "p2"], "heldout_standin": ["h1"]}))
    store = AnnotationStore(tmp_path / "ann")
    store.append([AnnotationRecord(doc_id=d, round=0 if d == "h1" else 1, status="blind" if d == "h1" else "reviewed",
                                   values=GOLD, source="human", annotator="local:test", backend="local")
                  for d in docs])
    overrides = [f"paths.{k}={tmp_path / v}" for k, v in {
        "registry": "registry.jsonl", "splits": "splits.yaml", "annotations": "ann", "prompts": "prompts",
        "runs": "runs", "review_queue": "rq"}.items()] + [
        f"cache.dir={tmp_path / 'cache' / 'llm'}", "gepa.max_metric_calls=20", "gepa.use_merge=false",
        "gepa.reflection_minibatch_size=2"]
    config = load_config(DEFAULT_YAML, overrides)
    fake = FakeModels()
    original = OpenRouterClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        original(self, *args, **kwargs)

    monkeypatch.setattr(OpenRouterClient, "__init__", patched)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    outcome = run_optimize(config, Settings(_env_file=None, openrouter_api_key="sk-or-x"), seed_candidate(), run_dir,
                           use_cache=False)

    assert FIX in outcome.best["field:garanties"]
    assert load_candidate(outcome.best_path) == outcome.best
    standin = outcome.heldout["heldout_standin"]
    assert standin["n"] == 1 and standin["scores"]["best"]["global_score"] == 1.0
    assert standin["scores"]["seed"]["global_score"] == pytest.approx(1 - 1 / 14)
    assert outcome.heldout["heldout_blind"]["n"] == 0
    for name in ("candidates.jsonl", "iterations.jsonl", "proposals.jsonl", "pareto_front.json", "result.json",
                 "heldout.json", "cost.json"):
        assert (run_dir / name).exists(), name

    # Held-out guard: h1's page never reaches the diagnostic or reflection model, and
    # it only reaches the extractor after GEPA has finished (held-out scoring).
    h1_url = load_pages(docs["h1"], config.render.dpi, 1)[0].content_part()["image_url"]["url"]
    mentions = [i for i, r in enumerate(fake.requests) if h1_url in json.dumps(r)]
    gepa_requests = [i for i, r in enumerate(fake.requests) if "attestation_auto" not in json.dumps(r.get("response_format"))]
    assert mentions and all(fake.requests[i]["model"] == config.role("extractor").model for i in mentions)
    assert min(mentions) > max(gepa_requests)

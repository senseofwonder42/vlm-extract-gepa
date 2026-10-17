"""Shared fixtures. No test here touches the network except `tests/test_live.py`."""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pymupdf
import pytest
from pydantic import SecretStr

from vex.config import Config, load_config
from vex.llm.client import OpenRouterClient
from vex.llm.cost import CostTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"
FAKE_KEY = SecretStr("sk-or-test-0000000000000000")

# Fictional certificate used by tests. Every name and number is invented.
FICTIONAL_LINES = [
    "ASSURANCES FICTIVES DU VAL",
    "Attestation d'assurance automobile",
    "Contrat n° FV-2025-004217",
    "Souscripteur : Camille Martin",
    "Adresse : 12 rue des Lilas, 69003 Lyon",
    "Véhicule : Peugeot 208 - Immatriculation AB-123-CD",
    "Valable du 01/02/2025 au 31/01/2026",
    "Code assureur : 4321",
    "Garanties : Responsabilité civile, Vol, Bris de glace",
]

VALID_VALUES = {
    "assureur": "Assurances Fictives du Val",
    "numero_contrat": "FV-2025-004217",
    "nom_souscripteur": "Camille Martin",
    "adresse_souscripteur": "12 rue des Lilas, 69003 Lyon",
    "immatriculation": "AB-123-CD",
    "marque_vehicule": "Peugeot",
    "date_debut_validite": "2025-02-01",
    "date_fin_validite": "2026-01-31",
    "code_assureur": "4321",
    "garanties": ["Responsabilité civile", "Vol", "Bris de glace"],
}


def make_pdf(path: Path, pages: int = 1) -> Path:
    """Write a small fictional certificate PDF with `pages` pages."""
    pdf = pymupdf.open()
    for index in range(pages):
        page = pdf.new_page()
        for line_no, line in enumerate(FICTIONAL_LINES):
            page.insert_text((50, 72 + 20 * line_no), f"{line} (p{index + 1})" if line_no == 0 else line)
    pdf.save(path)
    pdf.close()
    return path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Default config with every writable path redirected into `tmp_path`."""
    overrides = [f"cache.dir={tmp_path / 'cache'}", f"paths.runs={tmp_path / 'runs'}",
                 f"paths.prompts={tmp_path / 'prompts'}", "client.backoff_base_s=0", "client.backoff_max_s=0"]
    return load_config(DEFAULT_YAML, overrides)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    """A one-page fictional certificate PDF."""
    return make_pdf(tmp_path / "attestation.pdf")


def chat_response(content: str, cost: float = 0.001, reasoning_tokens: int = 0) -> dict:
    """Build an OpenRouter-shaped chat completion response."""
    return {
        "id": "gen-test",
        "provider": "Parasail",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def endpoints_response(structured: bool) -> dict:
    """Build an OpenRouter `/models/{id}/endpoints` response for one Parasail endpoint."""
    params = ["temperature", "top_k", "response_format", "reasoning"] + (["structured_outputs"] if structured else [])
    return {"data": {"endpoints": [{"tag": "parasail/fp8", "provider_name": "Parasail", "supported_parameters": params}]}}


class FakeOpenRouter:
    """Mock transport: records requests and replays queued chat responses.

    Each queued item is either a dict (returned with status 200) or a tuple
    `(status, body, headers)`.
    """

    def __init__(self, structured: bool = True) -> None:
        self.structured = structured
        self.queue: list[dict | tuple[int, dict, dict]] = []
        self.chat_requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json=endpoints_response(self.structured))
        self.chat_requests.append(json.loads(request.content))
        item = self.queue.pop(0)
        if isinstance(item, tuple):
            status, body, headers = item
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(200, json=item)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture
def fake_openrouter() -> FakeOpenRouter:
    """A fresh mock OpenRouter (structured outputs supported)."""
    return FakeOpenRouter()


@pytest.fixture
def make_client(config: Config, fake_openrouter: FakeOpenRouter) -> Callable[..., OpenRouterClient]:
    """Factory building a client wired to `fake_openrouter`."""

    def factory(cfg: Config | None = None, max_usd: float = 5.0, use_cache: bool = True) -> OpenRouterClient:
        return OpenRouterClient(
            cfg or config, FAKE_KEY, CostTracker(max_usd), use_cache=use_cache, transport=fake_openrouter.transport()
        )

    return factory

"""Opt-in live smoke test: ONE extractor call through OpenRouter.

Run with `uv run pytest -m live`. Needs OPENROUTER_API_KEY in `.env`.
Costs well under $0.01.
"""

import pytest

from vex.config import Settings
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.llm.cost import CostTracker
from vex.prompt import seed_candidate


@pytest.mark.live
async def test_one_live_extractor_call(config, pdf_path):
    key = Settings().openrouter_api_key
    if key is None or not key.get_secret_value():
        pytest.skip("OPENROUTER_API_KEY not set")
    tracker = CostTracker(max_usd=0.05)
    async with OpenRouterClient(config, key, tracker, use_cache=False) as client:
        result = await extract_document(client, pdf_path, seed_candidate())
    [call] = tracker.calls
    assert not result.failed, result.error
    assert result.values["immatriculation"] == "AB-123-CD"
    assert result.structured_path == "json_schema"
    assert call.reasoning_tokens == 0, "extractor reasoning must be disabled"
    assert 0 < call.usd < 0.01

"""OpenRouter client behaviour against a mocked transport."""

import asyncio
import json

import httpx
import pytest

from vex.llm.client import LLMError, OpenRouterClient
from vex.llm.cost import BudgetExceeded, CostTracker
from vex.schema import AttestationAuto, strict_json_schema
from tests.conftest import FAKE_KEY, VALID_VALUES, FakeOpenRouter, chat_response

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "doc"}]


async def test_params_and_extra_body_are_passed_through_unchanged(make_client, fake_openrouter, config):
    fake_openrouter.queue.append(chat_response("ok"))
    async with make_client() as client:
        await client.complete("extractor", MESSAGES, response_format={"type": "json_object"})
    [body] = fake_openrouter.chat_requests
    role = config.role("extractor")
    expected = {"model": role.model, "messages": MESSAGES, **role.params, **role.extra_body,
                "response_format": {"type": "json_object"}}
    assert body == expected


async def test_retries_on_429_and_5xx_then_succeeds(make_client, fake_openrouter):
    fake_openrouter.queue += [
        (429, {"error": {"message": "rate limited"}}, {"retry-after": "0"}),
        (503, {"error": {"message": "unavailable"}}, {}),
        {"error": {"code": 502, "message": "provider error"}},  # 200 with an error body
        chat_response("ok"),
    ]
    async with make_client() as client:
        completion = await client.complete("extractor", MESSAGES)
    assert completion.content == "ok"
    assert len(fake_openrouter.chat_requests) == 4


async def test_non_retryable_error_raises_immediately(make_client, fake_openrouter):
    fake_openrouter.queue.append((400, {"error": {"message": "bad request"}}, {}))
    async with make_client() as client:
        with pytest.raises(LLMError, match="400"):
            await client.complete("extractor", MESSAGES)
    assert len(fake_openrouter.chat_requests) == 1


async def test_gives_up_after_max_retries(make_client, fake_openrouter, config):
    attempts = config.client.max_retries + 1
    fake_openrouter.queue += [(500, {"error": {"message": "boom"}}, {})] * attempts
    async with make_client() as client:
        with pytest.raises(LLMError, match="Giving up"):
            await client.complete("extractor", MESSAGES)
    assert len(fake_openrouter.chat_requests) == attempts


async def test_backoff_is_exponential_and_honours_retry_after(config, monkeypatch):
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    fake = FakeOpenRouter()
    fake.queue += [(503, {}, {}), (503, {}, {}), (429, {}, {"retry-after": "30"}), chat_response("ok")]
    cfg = config.model_copy(update={"client": config.client.model_copy(update={"backoff_base_s": 1.0, "backoff_max_s": 60.0})})
    async with OpenRouterClient(cfg, FAKE_KEY, CostTracker(5), transport=fake.transport()) as client:
        await client.complete("extractor", MESSAGES)
    assert delays == [1.0, 2.0, 30.0]


async def test_semaphore_limits_concurrency(config):
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=chat_response("ok"))

    cfg = config.model_copy(update={"client": config.client.model_copy(update={"concurrency": 3})})
    async with OpenRouterClient(cfg, FAKE_KEY, CostTracker(5), use_cache=False,
                                transport=httpx.MockTransport(handler)) as client:
        await asyncio.gather(*(client.complete("extractor", [{"role": "user", "content": str(i)}]) for i in range(12)))
    assert peak == 3


async def test_cache_hit_skips_network_and_costs_nothing(make_client, fake_openrouter):
    fake_openrouter.queue += [chat_response(name, cost=0.01) for name in ("first", "second", "third")]
    async with make_client() as client:
        a = await client.complete("extractor", MESSAGES)
        b = await client.complete("extractor", MESSAGES)  # hit
        c = await client.complete("extractor", MESSAGES, use_cache=False)  # per-call bypass
        d = await client.complete("extractor", MESSAGES, cache_salt="sample-1")  # different key
    assert [x.content for x in (a, b, c, d)] == ["first", "first", "second", "third"]
    assert b.usage.cached and b.usage.usd == 0.0
    assert len(fake_openrouter.chat_requests) == 3
    assert client.tracker.total_usd == pytest.approx(0.03)


async def test_run_level_no_cache(make_client, fake_openrouter):
    fake_openrouter.queue += [chat_response("a"), chat_response("b")]
    async with make_client(use_cache=False) as client:
        await client.complete("extractor", MESSAGES)
        second = await client.complete("extractor", MESSAGES)
    assert second.content == "b"


async def test_budget_is_checked_before_sending(make_client, fake_openrouter):
    fake_openrouter.queue += [chat_response("a", cost=0.02), chat_response("b")]
    async with make_client(max_usd=0.01) as client:
        await client.complete("extractor", MESSAGES)
        with pytest.raises(BudgetExceeded):
            await client.complete("extractor", [{"role": "user", "content": "other"}])
    assert len(fake_openrouter.chat_requests) == 1


async def test_auto_uses_json_schema_when_supported(make_client, fake_openrouter):
    fake_openrouter.queue.append(chat_response(json.dumps(VALID_VALUES)))
    async with make_client() as client:
        result = await client.complete_json("extractor", MESSAGES, AttestationAuto, strict_json_schema())
    assert result.path == "json_schema" and not result.repaired
    assert result.parsed.immatriculation == "AB-123-CD"
    rf = fake_openrouter.chat_requests[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True


async def test_auto_falls_back_to_prompt_json(config):
    fake = FakeOpenRouter(structured=False)
    fake.queue.append(chat_response("```json\n" + json.dumps(VALID_VALUES) + "\n```"))
    async with OpenRouterClient(config, FAKE_KEY, CostTracker(5), transport=fake.transport()) as client:
        result = await client.complete_json("extractor", MESSAGES, AttestationAuto, strict_json_schema())
    assert result.path == "prompt" and result.parsed is not None
    body = fake.chat_requests[0]
    assert "response_format" not in body
    assert "Schéma JSON attendu" in body["messages"][0]["content"]


async def test_invalid_reply_gets_one_repair_retry(make_client, fake_openrouter):
    bad = {**VALID_VALUES, "date_debut_validite": "01/02/2025"}
    fake_openrouter.queue += [chat_response(json.dumps(bad)), chat_response(json.dumps(VALID_VALUES))]
    async with make_client() as client:
        result = await client.complete_json("extractor", MESSAGES, AttestationAuto, strict_json_schema())
    assert result.repaired and result.parsed is not None and result.error is None
    repair_messages = fake_openrouter.chat_requests[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": json.dumps(bad)}
    assert "ISO" in repair_messages[-1]["content"]
    assert len(result.usages) == 2


async def test_two_invalid_replies_give_no_parse(make_client, fake_openrouter):
    fake_openrouter.queue += [chat_response("not json"), chat_response("{still not json")]
    async with make_client() as client:
        result = await client.complete_json("extractor", MESSAGES, AttestationAuto, strict_json_schema())
    assert result.parsed is None and result.repaired
    assert result.raw_outputs == ["not json", "{still not json"]
    assert len(fake_openrouter.chat_requests) == 2


async def test_api_key_is_sent_only_as_header(make_client, fake_openrouter):
    fake_openrouter.queue.append(chat_response("ok"))
    async with make_client() as client:
        await client.complete("extractor", MESSAGES)
        assert client._http.headers["authorization"] == f"Bearer {FAKE_KEY.get_secret_value()}"
    assert FAKE_KEY.get_secret_value() not in json.dumps(fake_openrouter.chat_requests)

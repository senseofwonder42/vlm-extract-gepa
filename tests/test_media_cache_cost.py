"""Document rendering, cache keys and storage, cost accounting."""

import base64

import pytest

from vex.config import Price
from vex.llm.cache import DiskCache, cache_key
from vex.llm.cost import BudgetExceeded, CallUsage, CostTracker, usage_from_response
from vex.llm.media import load_pages
from vex.logs import redact
from tests.conftest import chat_response, make_pdf

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# --- media -------------------------------------------------------------------


def test_pdf_pages_are_rendered_as_png(tmp_path):
    pages = load_pages(make_pdf(tmp_path / "d.pdf", pages=2), dpi=72, max_pages=3)
    assert len(pages) == 2
    assert all(p.mime == "image/png" and p.data.startswith(b"\x89PNG") for p in pages)


def test_max_pages_caps_long_pdfs(tmp_path):
    assert len(load_pages(make_pdf(tmp_path / "d.pdf", pages=5), dpi=72, max_pages=3)) == 3


def test_dpi_changes_resolution(tmp_path):
    pdf = make_pdf(tmp_path / "d.pdf")
    assert len(load_pages(pdf, dpi=150, max_pages=1)[0].data) > len(load_pages(pdf, dpi=50, max_pages=1)[0].data)


def test_images_are_passed_through_unchanged(tmp_path):
    path = tmp_path / "scan.png"
    path.write_bytes(PNG_1PX)
    [page] = load_pages(path, dpi=150, max_pages=3)
    assert page.data == PNG_1PX
    assert page.content_part()["image_url"]["url"].startswith("data:image/png;base64,")


def test_unsupported_type_is_rejected(tmp_path):
    path = tmp_path / "doc.docx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_pages(path, dpi=150, max_pages=3)


# --- cache -------------------------------------------------------------------


def _payload(image_url: str, temperature: float = 0.0) -> dict:
    content = [{"type": "text", "text": "doc"}, {"type": "image_url", "image_url": {"url": image_url}}]
    return {"model": "m", "temperature": temperature, "messages": [{"role": "user", "content": content}]}


def test_cache_key_depends_on_image_content_params_and_salt():
    base = cache_key(_payload("data:image/png;base64,AAAA"))
    assert base == cache_key(_payload("data:image/png;base64,AAAA"))
    assert base != cache_key(_payload("data:image/png;base64,BBBB"))
    assert base != cache_key(_payload("data:image/png;base64,AAAA", temperature=0.5))
    assert base != cache_key({**_payload("data:image/png;base64,AAAA"), "_salt": "sample-1"})


def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("ab" * 32) is None
    cache.set("ab" * 32, {"x": 1})
    assert cache.get("ab" * 32) == {"x": 1}
    assert not list(tmp_path.rglob("*.tmp"))


# --- cost --------------------------------------------------------------------

PRICES = {"m": Price(input=1.0, output=10.0)}


def test_usage_prefers_reported_cost():
    usage = usage_from_response(chat_response("x", cost=0.0042, reasoning_tokens=7), "extractor", "m", PRICES, False)
    assert usage.usd == 0.0042
    assert usage.reasoning_tokens == 7


def test_usage_falls_back_to_price_table():
    response = chat_response("x")
    del response["usage"]["cost"]
    usage = usage_from_response(response, "extractor", "m", PRICES, False)
    assert usage.usd == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6)


def test_cached_calls_cost_nothing():
    assert usage_from_response(chat_response("x", cost=0.5), "extractor", "m", PRICES, True).usd == 0.0


def test_budget_guard_and_summary():
    tracker = CostTracker(max_usd=0.01)
    tracker.check_budget()
    tracker.record(CallUsage("extractor", "m", 10, 5, 0, 0.006, False))
    tracker.record(CallUsage("diagnostic", "h", 10, 5, 3, 0.005, False))
    with pytest.raises(BudgetExceeded):
        tracker.check_budget()
    summary = tracker.summary()
    assert summary["calls"] == 2
    assert summary["by_role"]["diagnostic"]["reasoning_tokens"] == 3


def test_redact_masks_keys():
    assert redact("auth Bearer sk-or-v1-abcdef123456 failed") == "auth *** failed"
    assert "sk-or" not in redact("key=sk-or-v1-abcdefghijk")

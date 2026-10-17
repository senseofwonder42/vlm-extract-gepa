"""Specimen fetcher against a mocked web: approval, robots.txt, PII, manifest."""

import hashlib

import httpx
import yaml

from vex.data.specimens import USER_AGENT, fetch_specimens, load_manifest

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def _source(id_: str, url: str, status: str = "approved", pii: str = "none") -> dict:
    return {"id": id_, "url": url, "page_url": url, "source": "test", "licence_note": "test",
            "status": status, "pii": pii}


def _web(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        if request.url.host == "blocked.example":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(404)
    if request.url.path.endswith(".html"):
        return httpx.Response(200, text="<html/>", headers={"content-type": "text/html"})
    return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})


def test_fetch_respects_status_pii_robots_and_type(tmp_path):
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({"sources": [
        _source("ok", "https://ok.example/a.jpg"),
        _source("proposed", "https://ok.example/b.jpg", status="proposed"),
        _source("excluded", "https://ok.example/c.jpg", status="excluded"),
        _source("pii", "https://ok.example/d.jpg", pii="suspected"),
        _source("robots", "https://blocked.example/e.jpg"),
        _source("html", "https://ok.example/page.html"),
    ]}))
    manifest = tmp_path / "manifest.jsonl"
    with httpx.Client(transport=httpx.MockTransport(_web), headers={"User-Agent": USER_AGENT}) as client:
        new = fetch_specimens(sources, tmp_path / "real", manifest, client, delay_s=0)
        again = fetch_specimens(sources, tmp_path / "real", manifest, client, delay_s=0)

    assert [entry.id for entry in new] == ["ok"]
    assert again == []
    [entry] = load_manifest(manifest)
    assert entry.file == "ok.jpg" and entry.sha256 == hashlib.sha256(JPEG).hexdigest()
    assert (tmp_path / "real" / "ok.jpg").read_bytes() == JPEG
    assert sorted(p.name for p in (tmp_path / "real").iterdir()) == ["ok.jpg"]


def test_committed_source_list_is_valid_and_excludes_suspected_pii():
    from pathlib import Path

    from vex.data.specimens import load_sources

    sources = load_sources(Path(__file__).resolve().parents[1] / "data" / "specimen_sources.yaml")
    assert sources
    assert all(s.status != "approved" for s in sources if s.pii == "suspected")
    assert len({s.id for s in sources}) == len(sources)

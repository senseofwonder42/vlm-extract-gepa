"""Download real certificate specimens listed in `data/specimen_sources.yaml`.

The source list is curated by hand and reviewed by a person: each entry has a
`status` (`proposed`, `approved`, `excluded`) and a `pii` assessment
(`none`, `redacted`, `suspected`). Only `approved` entries are downloaded, and
an entry marked `pii: suspected` is refused even if approved. Before each
download the site's robots.txt is checked for our user agent, and requests
are spaced by `delay_s`.

Every downloaded file is recorded in `data/manifest.jsonl` (url, sha256,
retrieval date, source and licence note), so the dataset can be rebuilt; the
files themselves stay in the git-ignored `data/raw/real/`.
"""

import hashlib
import time
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
import yaml
from loguru import logger
from pydantic import BaseModel

USER_AGENT = "vlm-extract-gepa/0.1 (research dataset; contact: repository owner)"
_SUFFIX_BY_TYPE = {"application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg"}


class SpecimenSource(BaseModel):
    """One candidate specimen.

    Attributes:
        id: Short stable identifier, used as file name.
        url: Direct URL of the file.
        page_url: Page where the file is published.
        source: Publisher.
        licence_note: Licence or usage basis.
        status: Review status.
        pii: Personal-data assessment.
        notes: Free text (what the document shows, why excluded...).
    """

    id: str
    url: str
    page_url: str
    source: str
    licence_note: str
    status: Literal["proposed", "approved", "excluded"]
    pii: Literal["none", "redacted", "suspected"]
    notes: str = ""


class ManifestEntry(BaseModel):
    """One downloaded specimen, as recorded in `manifest.jsonl`."""

    id: str
    url: str
    page_url: str
    file: str
    sha256: str
    retrieved_at: str
    source: str
    licence_note: str
    pii: str


def load_sources(path: Path) -> list[SpecimenSource]:
    """Load the curated source list.

    Args:
        path: `data/specimen_sources.yaml`.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [SpecimenSource.model_validate(item) for item in data.get("sources", [])]


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Load the manifest (empty if missing).

    Args:
        path: `data/manifest.jsonl`.
    """
    if not path.exists():
        return []
    return [ManifestEntry.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def robots_allows(client: httpx.Client, url: str) -> bool:
    """Check robots.txt for `USER_AGENT`. An unreachable robots.txt means "allowed".

    Args:
        client: HTTP client.
        url: URL to fetch.

    Returns:
        True if fetching `url` is allowed.
    """
    parts = urlparse(url)
    parser = urllib.robotparser.RobotFileParser()
    try:
        response = client.get(f"{parts.scheme}://{parts.netloc}/robots.txt")
    except httpx.HTTPError:
        return True
    if response.status_code >= 400:
        return True
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def fetch_specimens(
    sources_path: Path, out_dir: Path, manifest_path: Path, client: httpx.Client, delay_s: float = 2.0
) -> list[ManifestEntry]:
    """Download every approved specimen not already in the manifest.

    Args:
        sources_path: `data/specimen_sources.yaml`.
        out_dir: `data/raw/real`.
        manifest_path: `data/manifest.jsonl` (appended to).
        client: HTTP client (its user agent should be `USER_AGENT`).
        delay_s: Pause between two downloads.

    Returns:
        The new manifest entries.
    """
    done = {entry.id for entry in load_manifest(manifest_path) if (out_dir / entry.file).exists()}
    out_dir.mkdir(parents=True, exist_ok=True)
    new: list[ManifestEntry] = []
    for source in load_sources(sources_path):
        if source.status != "approved" or source.id in done:
            continue
        if source.pii == "suspected":
            logger.warning("{}: approved but pii=suspected, refusing to download", source.id)
            continue
        if not robots_allows(client, source.url):
            logger.warning("{}: disallowed by robots.txt, skipped", source.id)
            continue
        if new:
            time.sleep(delay_s)
        response = client.get(source.url)
        if response.status_code != 200:
            logger.warning("{}: HTTP {}, skipped", source.id, response.status_code)
            continue
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        suffix = _SUFFIX_BY_TYPE.get(content_type)
        if suffix is None:
            logger.warning("{}: unsupported content type {!r}, skipped", source.id, content_type)
            continue
        path = out_dir / f"{source.id}{suffix}"
        path.write_bytes(response.content)
        entry = ManifestEntry(
            id=source.id, url=source.url, page_url=source.page_url, file=path.name,
            sha256=hashlib.sha256(response.content).hexdigest(),
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source=source.source, licence_note=source.licence_note, pii=source.pii,
        )
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        new.append(entry)
        logger.info("Downloaded {} ({} bytes)", path.name, len(response.content))
    return new

"""Disk cache for LLM responses.

The key is the sha256 of the canonical JSON of everything that determines the
answer: model, params, extra_body, response_format and messages. Images inside
messages are replaced by the sha256 of their data URL before hashing, so the key
stays small and two identical images give the same key whatever their path.

Entries are plain JSON files under `<dir>/<key[:2]>/<key>.json`, written
atomically, so a crashed run never leaves a half-written entry.
"""

import hashlib
import json
import os
from pathlib import Path

from pydantic import JsonValue


def _hash_images(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            url = value["image_url"]["url"]  # type: ignore[index]
            return {"type": "image_url", "sha256": hashlib.sha256(str(url).encode()).hexdigest()}
        return {key: _hash_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hash_images(item) for item in value]
    return value


def cache_key(payload: dict[str, JsonValue]) -> str:
    """Compute the cache key of a request payload.

    Args:
        payload: The request body (plus any salt such as a sample index).

    Returns:
        Hex sha256 digest.
    """
    canonical = json.dumps(_hash_images(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """A directory of JSON responses keyed by `cache_key`."""

    def __init__(self, directory: Path) -> None:
        """Create the cache.

        Args:
            directory: Root directory; created on first write.
        """
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """Return the cached response, or `None` on a miss.

        Args:
            key: Cache key.
        """
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict) -> None:
        """Store a response atomically.

        Args:
            key: Cache key.
            value: JSON-serializable response.
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

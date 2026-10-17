"""Loguru setup with secret redaction.

Every sink goes through `_redact`, which masks anything that looks like an
OpenRouter key. The code never logs secrets on purpose; this is a safety net
for exception messages or response bodies that might echo one back.
"""

import re
import sys
from pathlib import Path

from loguru import logger

_SECRET_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-.]{8,}")


def redact(text: str) -> str:
    """Mask API keys and bearer tokens in a string.

    Args:
        text: Any text that might contain a secret.

    Returns:
        The text with secrets replaced by `***`.
    """
    return _SECRET_PATTERN.sub("***", text)


def _patch(record: dict) -> None:
    record["message"] = redact(record["message"])


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure loguru: stderr sink, optional file sink, redaction on both.

    Args:
        level: Minimum level for stderr.
        log_file: If given, also log at DEBUG level to this file.
    """
    logger.remove()
    logger.configure(patcher=_patch)
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> {message}")
    if log_file is not None:
        logger.add(log_file, level="DEBUG", encoding="utf-8")

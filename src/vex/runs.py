"""Run directories: one per command invocation that calls a model.

Layout::

    runs/<UTC timestamp>-<command>-<hash8>/
        config.resolved.yaml   # the exact config used, after --set overrides
        config.sha256          # its hash
        log.txt                # DEBUG log of the run
        cost.json              # written by the command at the end
"""

from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger

from vex.config import Config, config_hash
from vex.logs import setup_logging


def create_run_dir(config: Config, command: str) -> Path:
    """Create a run directory, write the resolved config and start file logging.

    Args:
        config: Resolved configuration of this run.
        command: Short command name used in the directory name (e.g. `extract`).

    Returns:
        Path of the new run directory.
    """
    digest = config_hash(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.paths.runs / f"{stamp}-{command}-{digest[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    (run_dir / "config.resolved.yaml").write_text(resolved, encoding="utf-8")
    (run_dir / "config.sha256").write_text(digest + "\n", encoding="utf-8")
    setup_logging(log_file=run_dir / "log.txt")
    logger.info("Run directory: {} (config {})", run_dir, digest[:8])
    return run_dir

"""Frozen, seeded split assignment (docs/PLAN.md §6).

`data/splits.yaml` assigns every registered document once:

- `heldout_blind`: REAL documents only, annotated blind, never given to GEPA,
  the reflection model or the bootstrap rewriter;
- `heldout_standin` (Mac only, `splits.standin_layouts`): synthetic documents of
  reserved layouts, annotated blind and held out like `heldout_blind`, but
  reported separately. It measures generalization to an unseen layout, not
  real-world accuracy. Other documents of those layouts go to
  `reserved_unused`, so a reserved layout never reaches GEPA;
- `gepa_train` / `gepa_pareto`: *buckets*. A document in a bucket becomes part
  of GEPA's trainset / valset only once it has a gold annotation (that
  intersection is computed by the annotation store, P4).

Assignment is incremental: rerunning after new documents arrive keeps every
existing assignment and places only the new ones. A document's placement
depends only on `sha256(seed, doc_id)`, not on the order files were added.
The held-out target is half of the real documents, capped at
`heldout_blind_size`. Below `MIN_HELDOUT` a warning says the held-out score
is low-confidence.
"""

import hashlib
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel

from vex.config import SplitsConfig
from vex.data.registry import DocRecord

MIN_HELDOUT = 8


class Splits(BaseModel):
    """Split assignment, as stored in `data/splits.yaml`.

    Attributes:
        seed: Seed used for every assignment.
        heldout_blind: Real documents held out for blind evaluation.
        heldout_standin: Synthetic documents of reserved layouts, held out (Mac stand-in).
        reserved_unused: Other documents of reserved layouts; never used.
        gepa_train: Bucket feeding GEPA's trainset (once reviewed).
        gepa_pareto: Bucket feeding GEPA's valset (once reviewed).
    """

    seed: int
    heldout_blind: list[str] = []
    heldout_standin: list[str] = []
    reserved_unused: list[str] = []
    gepa_train: list[str] = []
    gepa_pareto: list[str] = []

    def assigned(self) -> set[str]:
        """Return every assigned doc_id."""
        return {*self.heldout_blind, *self.heldout_standin, *self.reserved_unused, *self.gepa_train, *self.gepa_pareto}


def _rank(seed: int, purpose: str, doc_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{purpose}:{doc_id}".encode()).hexdigest()
    return int(digest, 16) / 16**64


def assign_splits(registry: list[DocRecord], existing: Splits | None, config: SplitsConfig) -> Splits:
    """Place every unassigned document, keeping existing assignments.

    Args:
        registry: All registered documents.
        existing: Current splits, or `None` for a first run.
        config: Split settings.

    Returns:
        The updated splits.

    Raises:
        ValueError: If `existing` was created with a different seed.
    """
    splits = existing.model_copy(deep=True) if existing else Splits(seed=config.seed)
    if splits.seed != config.seed:
        raise ValueError(f"splits.yaml uses seed {splits.seed}, config says {config.seed}; refusing to mix")
    assigned = splits.assigned()
    new = [record for record in registry if record.doc_id not in assigned]

    n_real = sum(record.origin == "real" for record in registry)
    target = min(config.heldout_blind_size, n_real // 2)
    new_real = sorted((r for r in new if r.origin == "real"), key=lambda r: _rank(splits.seed, "heldout", r.doc_id))
    held = set()
    for record in new_real:
        if len(splits.heldout_blind) >= target:
            break
        splits.heldout_blind.append(record.doc_id)
        held.add(record.doc_id)

    reserved = sorted(
        (r for r in new if r.origin == "synthetic" and r.layout in config.standin_layouts),
        key=lambda r: _rank(splits.seed, "standin", r.doc_id),
    )
    for record in reserved:
        target_list = splits.heldout_standin if len(splits.heldout_standin) < config.heldout_standin_size else splits.reserved_unused
        target_list.append(record.doc_id)
        held.add(record.doc_id)

    for record in new:
        if record.doc_id in held:
            continue
        bucket = splits.gepa_pareto if _rank(splits.seed, "bucket", record.doc_id) < config.pareto_fraction else splits.gepa_train
        bucket.append(record.doc_id)

    if len(splits.heldout_blind) < MIN_HELDOUT:
        logger.warning(
            "heldout_blind has {} real document(s) (< {}): held-out scores will be low-confidence",
            len(splits.heldout_blind), MIN_HELDOUT,
        )
    logger.info("Splits: heldout_blind={} heldout_standin={} reserved_unused={} gepa_train={} gepa_pareto={} (+{} new)",
                len(splits.heldout_blind), len(splits.heldout_standin), len(splits.reserved_unused),
                len(splits.gepa_train), len(splits.gepa_pareto), len(new))
    return splits


def load_splits(path: Path) -> Splits | None:
    """Load `splits.yaml`, or `None` if it does not exist yet.

    Args:
        path: `data/splits.yaml`.
    """
    if not path.exists():
        return None
    return Splits.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_splits(splits: Splits, path: Path) -> None:
    """Write `splits.yaml`.

    Args:
        splits: Splits to save.
        path: Destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Frozen split assignment. Generated by `vex data split`; do not edit by hand.\n"
    path.write_text(header + yaml.safe_dump(splits.model_dump(), sort_keys=False), encoding="utf-8")

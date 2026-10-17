"""Document registry and frozen split assignment."""

import random

import pytest

from vex.config import SplitsConfig
from vex.data.registry import DocRecord, file_sha256, load_registry, register
from vex.data.splits import assign_splits, load_splits, save_splits
from tests.conftest import make_pdf

CONFIG = SplitsConfig(seed=42, heldout_blind_size=15, pareto_fraction=0.4)


def test_register_records_origin_hash_pages_and_skips_duplicates(tmp_path):
    raw = tmp_path / "raw"
    (raw / "real").mkdir(parents=True)
    (raw / "synthetic").mkdir()
    make_pdf(raw / "real" / "a.pdf", pages=2)
    (raw / "synthetic" / "b.png").write_bytes(b"\x89PNG fake")
    (raw / "synthetic" / "copy_of_b.png").write_bytes(b"\x89PNG fake")  # same content
    (raw / "synthetic" / "notes.txt").write_text("ignored")
    registry_path = tmp_path / "registry.jsonl"

    new = register(raw, registry_path)
    assert len(new) == 2
    by_origin = {r.origin: r for r in new}
    assert by_origin["real"].pages == 2
    assert by_origin["real"].doc_id == file_sha256(raw / "real" / "a.pdf")
    assert register(raw, registry_path) == []
    assert len(load_registry(registry_path)) == 2


def _records(n_real: int, n_synth: int) -> list[DocRecord]:
    def rec(i: int, origin: str) -> DocRecord:
        return DocRecord(doc_id=f"{origin}{i:04d}".ljust(64, "0"), origin=origin, path=f"{origin}/{i}", pages=1,
                         added_at="2026-10-17T00:00:00+00:00")
    return [rec(i, "real") for i in range(n_real)] + [rec(i, "synthetic") for i in range(n_synth)]


def test_heldout_is_real_only_half_of_real_and_capped():
    registry = _records(10, 100)
    splits = assign_splits(registry, None, CONFIG)
    real_ids = {r.doc_id for r in registry if r.origin == "real"}
    assert len(splits.heldout_blind) == 5
    assert set(splits.heldout_blind) <= real_ids
    big = assign_splits(_records(60, 10), None, CONFIG)
    assert len(big.heldout_blind) == 15


def test_every_doc_assigned_exactly_once():
    registry = _records(10, 100)
    splits = assign_splits(registry, None, CONFIG)
    all_ids = splits.heldout_blind + splits.gepa_train + splits.gepa_pareto
    assert sorted(all_ids) == sorted(r.doc_id for r in registry)
    assert 0.25 < len(splits.gepa_pareto) / (len(splits.gepa_train) + len(splits.gepa_pareto)) < 0.55


def test_assignment_is_frozen_and_incremental():
    first = assign_splits(_records(4, 50), None, CONFIG)
    second = assign_splits(_records(20, 80), first, CONFIG)
    for name in ("heldout_blind", "gepa_train", "gepa_pareto"):
        assert getattr(second, name)[: len(getattr(first, name))] == getattr(first, name)
    assert len(second.heldout_blind) == 10  # topped up to half of the 20 real docs
    assert assign_splits(_records(20, 80), second, CONFIG) == second


def test_assignment_does_not_depend_on_registry_order():
    registry = _records(12, 60)
    shuffled = registry[:]
    random.Random(0).shuffle(shuffled)
    a, b = assign_splits(registry, None, CONFIG), assign_splits(shuffled, None, CONFIG)
    assert set(a.heldout_blind) == set(b.heldout_blind) and set(a.gepa_pareto) == set(b.gepa_pareto)


def test_seed_mismatch_is_refused():
    splits = assign_splits(_records(4, 4), None, CONFIG)
    with pytest.raises(ValueError, match="seed"):
        assign_splits(_records(4, 4), splits, SplitsConfig(seed=7, heldout_blind_size=15, pareto_fraction=0.4))


def test_splits_round_trip(tmp_path):
    splits = assign_splits(_records(6, 20), None, CONFIG)
    save_splits(splits, tmp_path / "splits.yaml")
    assert load_splits(tmp_path / "splits.yaml") == splits
    assert load_splits(tmp_path / "missing.yaml") is None


def _with_layouts(records: list[DocRecord]) -> list[DocRecord]:
    layouts = ("memo", "carte_verte", "attestation_lettre")
    return [r.model_copy(update={"layout": layouts[i % 3]}) if r.origin == "synthetic" else r
            for i, r in enumerate(records)]


def test_standin_layout_never_reaches_gepa_buckets():
    config = SplitsConfig(seed=42, heldout_blind_size=15, pareto_fraction=0.4,
                          standin_layouts=["carte_verte"], heldout_standin_size=10)
    registry = _with_layouts(_records(3, 90))
    splits = assign_splits(registry, None, config)
    reserved = {r.doc_id for r in registry if r.layout == "carte_verte"}
    assert len(reserved) == 30
    assert len(splits.heldout_standin) == 10
    assert set(splits.heldout_standin) | set(splits.reserved_unused) == reserved
    assert not reserved & (set(splits.gepa_train) | set(splits.gepa_pareto))
    assert all(r.origin == "real" for r in registry if r.doc_id in splits.heldout_blind)


def test_register_reads_synthetic_layout_from_file_name(tmp_path):
    raw = tmp_path / "raw"
    (raw / "synthetic").mkdir(parents=True)
    (raw / "synthetic" / "syn_0001_carte_verte.jpg").write_bytes(b"a")
    (raw / "synthetic" / "other.jpg").write_bytes(b"b")
    layouts = {r.path.split("/")[-1]: r.layout for r in register(raw, tmp_path / "registry.jsonl")}
    assert layouts == {"syn_0001_carte_verte.jpg": "carte_verte", "other.jpg": None}

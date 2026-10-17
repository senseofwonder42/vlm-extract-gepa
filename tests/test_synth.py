"""Synthetic generator: determinism, layouts, fictional insurers, meta consistency."""

import hashlib
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

from vex.data.synth.augment import augment, sample_params
from vex.data.synth.generator import generate, generate_one, render_pdf
from vex.data.synth.values import FICTIONAL_INSURERS, LAYOUTS, sample_document
from vex.schema import AttestationAuto

SRC = Path(__file__).resolve().parents[1] / "src" / "vex"

# Real French motor insurers and brands: no fictional name may contain one.
REAL_INSURERS = (
    "maif", "macif", "matmut", "axa", "allianz", "groupama", "gmf", "maaf", "generali", "covea", "aviva",
    "pacifica", "direct assurance", "leocare", "ornikar", "mma", "gan", "macsf", "amv", "april", "olivier",
    "euro-assurance", "lovys", "credit agricole", "crédit agricole", "lcl", "bnp", "societe generale",
    "société générale", "cic", "credit mutuel", "crédit mutuel", "banque postale", "aon", "hiscox", "zurich",
    "mutuelle de poitiers", "thelem", "monceau", "ag2r", "mgen", "swisslife", "swiss life",
)


def test_insurers_are_fictional():
    for insurer in FICTIONAL_INSURERS:
        words = re.findall(r"[a-zà-ÿ'-]+", insurer.name.lower())
        joined = " ".join(words)
        for real in REAL_INSURERS:
            assert not re.search(rf"\b{re.escape(real)}\b", joined), (insurer.name, real)


def test_layouts_cycle_and_there_are_at_least_five():
    assert len(LAYOUTS) >= 5
    assert [sample_document(i, 1).layout for i in range(len(LAYOUTS))] == list(LAYOUTS)


def test_sampling_is_deterministic():
    assert sample_document(7, 42) == sample_document(7, 42)
    assert sample_document(7, 42) != sample_document(7, 43)


@pytest.mark.parametrize("index", range(15))
def test_meta_validates_against_schema(index):
    meta = sample_document(index, 42).meta
    assert AttestationAuto.model_validate(meta).model_dump() == meta


def _pdf_text(sample) -> str:
    with pymupdf.open("pdf", render_pdf(sample)) as pdf:
        return " ".join(page.get_text() for page in pdf)


def _squash(text: str) -> str:
    # NFKC folds typographic ligatures ("ﬀ" in "Goﬀ") that PDF text extraction returns.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).lower()


@pytest.mark.parametrize("index", range(200))
def test_every_meta_value_is_printed(index):
    """Sanity check of synthetic_meta (tests may read it): what meta claims is visible."""
    sample = sample_document(index, 42)
    text = _squash(_pdf_text(sample))
    meta, display = sample.meta, sample.display
    assert _squash(meta["numero_contrat"]) in text
    assert _squash(meta["nom_souscripteur"]) in text
    assert meta["marque_vehicule"].lower() in text
    if meta["immatriculation"]:
        assert _squash(display["vehicles"][0]["plate_display"]) in text
    if meta["code_assureur"]:
        assert meta["code_assureur"] in text
    if meta["garanties"]:
        for guarantee in meta["garanties"]:
            assert guarantee.lower() in text
    if meta["adresse_souscripteur"]:
        assert _squash(display["street"]) in text and display["postcode"] in text


def test_hidden_fields_are_not_printed():
    for index in range(40):
        sample = sample_document(index, 42)
        text = _squash(_pdf_text(sample))
        if sample.meta["code_assureur"] is None:
            assert "code assureur" not in text, index
        if sample.meta["adresse_souscripteur"] is None:
            assert _squash(sample.display["street"]) not in text, index


def test_two_page_documents_render_two_pages():
    two_page = next(sample_document(i, 42) for i in range(200) if sample_document(i, 42).pages == 2)
    with pymupdf.open("pdf", render_pdf(two_page)) as pdf:
        assert pdf.page_count == 2


def test_some_documents_have_two_vehicles_and_only_the_first_is_expected():
    sample = next(sample_document(i, 42) for i in range(200) if len(sample_document(i, 42).display["vehicles"]) == 2)
    assert sample.meta["immatriculation"] == sample.display["vehicles"][0]["plate"]


def test_render_is_byte_deterministic():
    sample = sample_document(3, 42)
    assert render_pdf(sample) == render_pdf(sample)


def test_augmentation_parameters_stay_in_bounds():
    rng = np.random.default_rng(0)
    for _ in range(300):
        params = sample_params(rng)
        assert params.angle in (90.0, 180.0, 270.0) or -4.0 <= params.angle <= 4.0
        assert 0 <= params.blur_radius <= 1.3 and 0.7 <= params.contrast <= 1.25
        assert 30 <= params.jpeg_quality <= 85


def test_augment_returns_decodable_jpeg_and_rotates():
    page = Image.new("RGB", (200, 100), "white")
    params = sample_params(np.random.default_rng(1))
    params = params.__class__(**{**params.as_dict(), "angle": 90.0})
    data = augment(page, params, np.random.default_rng(1))
    image = Image.open(io.BytesIO(data))
    assert image.format == "JPEG" and image.size == (100, 200)


def test_generate_is_reproducible_and_guards_overwrite(tmp_path):
    a = generate(6, 42, tmp_path / "a", tmp_path / "a_meta.jsonl")
    b = generate(6, 42, tmp_path / "b", tmp_path / "b_meta.jsonl")
    digest = lambda files: [hashlib.sha256(f.path.read_bytes()).hexdigest() for f in files]  # noqa: E731
    assert digest(a) == digest(b)
    assert len((tmp_path / "a_meta.jsonl").read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        generate(6, 42, tmp_path / "a", tmp_path / "a_meta.jsonl")
    assert len(generate(3, 42, tmp_path / "a", tmp_path / "a_meta.jsonl", overwrite=True)) == 3
    assert len(list((tmp_path / "a").glob("syn_*"))) == 3


def test_scan_fraction_extremes(tmp_path):
    assert generate_one(0, 42, tmp_path, scan_fraction=0.0).path.suffix == ".pdf"
    scanned = generate_one(1, 42, tmp_path, scan_fraction=1.0)
    assert scanned.meta["scan"] is not None


def test_synthetic_meta_is_only_touched_by_the_generator():
    """synthetic_meta must never become labels: only `vex data synth` touches its path."""
    users = sorted(p.name for p in SRC.rglob("*.py") if ".synthetic_meta" in p.read_text())
    assert users == ["cli.py"]
    cli_lines = [line for line in (SRC / "cli.py").read_text().splitlines() if ".synthetic_meta" in line]
    assert len(cli_lines) == 1 and "generate(" in cli_lines[0]


def test_generation_is_identical_across_processes(tmp_path):
    """Regression: Pillow's JPEG encoder was nondeterministic across processes.

    In-process tests cannot see that, so generate the same scans in separate
    interpreters and compare bytes.
    """
    import subprocess
    import sys

    script = (
        "import hashlib, sys; from pathlib import Path; from vex.data.synth.generator import generate_one; "
        "d = Path(sys.argv[1]); d.mkdir(); "
        "print(' '.join(hashlib.sha256(generate_one(i, 42, d, scan_fraction=1.0).path.read_bytes()).hexdigest()[:8] "
        "for i in range(12)))"
    )
    outputs = {
        subprocess.run([sys.executable, "-c", script, str(tmp_path / f"run{k}")], capture_output=True, text=True,
                       check=True).stdout
        for k in range(4)
    }
    assert len(outputs) == 1, outputs

"""Candidate building, rendering, validation, versioned storage."""

import json

import pytest

from vex.prompt import (
    INSTRUCTIONS_KEY,
    candidate_hash,
    component_key,
    load_candidate,
    render,
    save_candidate,
    seed_candidate,
    validate_candidate,
)
from vex.schema import FIELD_NAMES, field_descriptions


def test_seed_candidate_has_one_component_per_field_plus_instructions():
    candidate = seed_candidate()
    assert set(candidate) == {INSTRUCTIONS_KEY, *(f"field:{name}" for name in FIELD_NAMES)}
    assert candidate["field:immatriculation"] == field_descriptions()["immatriculation"]


def test_render_puts_candidate_texts_in_prompt_and_schema():
    candidate = seed_candidate()
    candidate[component_key("assureur")] = "TEXTE MODIFIÉ PAR GEPA"
    rendered = render(candidate)
    assert rendered.system.startswith(candidate[INSTRUCTIONS_KEY])
    assert "- assureur : TEXTE MODIFIÉ PAR GEPA" in rendered.system
    assert rendered.json_schema["properties"]["assureur"]["description"] == "TEXTE MODIFIÉ PAR GEPA"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("field:assureur"),
        lambda c: c.update({"field:vin": "x"}),
        lambda c: c.update({"instructions": "   "}),
    ],
)
def test_invalid_candidates_are_rejected(mutate):
    candidate = seed_candidate()
    mutate(candidate)
    with pytest.raises(ValueError):
        validate_candidate(candidate)


def test_hash_is_content_based():
    a, b = seed_candidate(), seed_candidate()
    assert candidate_hash(a) == candidate_hash(b)
    b["instructions"] += " "
    assert candidate_hash(a) != candidate_hash(b)


def test_save_and_load_round_trip(tmp_path):
    candidate = seed_candidate()
    path = save_candidate(candidate, tmp_path, "seed")
    assert path.name == f"seed.{candidate_hash(candidate)[:8]}.json"
    assert load_candidate(path) == candidate


def test_hand_edited_file_is_detected(tmp_path):
    path = save_candidate(seed_candidate(), tmp_path, "seed")
    payload = json.loads(path.read_text())
    payload["candidate"]["instructions"] = "edited"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        load_candidate(path)

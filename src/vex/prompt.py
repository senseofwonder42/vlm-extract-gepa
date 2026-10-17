"""The extraction prompt as a GEPA candidate.

A candidate is a flat `dict[str, str]`, exactly what `gepa.optimize` evolves::

    {
        "instructions": "<global extraction instructions>",
        "field:assureur": "<description of the assureur field>",
        ...
        "field:garanties": "<description of the garanties field>",
    }

Each key is one *component*: GEPA rewrites one or a few components per
iteration. `render` turns a candidate into what the model actually receives:
a system prompt (instructions + field list) and a JSON schema whose field
descriptions are the candidate's texts.

Candidates are saved as `prompts/<name>.<hash8>.json`; the hash covers the
candidate content only, so the same text always gets the same hash.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vex.schema import FIELD_NAMES, field_descriptions, strict_json_schema

INSTRUCTIONS_KEY = "instructions"
FIELD_PREFIX = "field:"

SEED_INSTRUCTIONS = (
    "Tu extrais des champs d'un justificatif d'assurance automobile français : attestation "
    "d'assurance, Mémo Véhicule Assuré, carte internationale d'assurance automobile (blanche) "
    "ou ancienne carte verte. Les images sont les pages du document, éventuellement scannées, "
    "tournées ou de mauvaise qualité ; lis-les entièrement avant de répondre.\n"
    "Pour chaque champ, suis sa description. Recopie les valeurs telles qu'elles apparaissent "
    "dans le document, sans rien inventer ni compléter. Si un champ n'apparaît pas ou est "
    "illisible, renvoie null, jamais une chaîne vide.\n"
    "Réponds uniquement avec un objet JSON contenant exactement les champs demandés."
)

Candidate = dict[str, str]


@dataclass(frozen=True)
class RenderedPrompt:
    """What the model receives for one candidate.

    Attributes:
        system: System prompt text.
        json_schema: Strict JSON schema with the candidate's field descriptions.
    """

    system: str
    json_schema: dict


def component_key(field: str) -> str:
    """Return the candidate key of a field component (e.g. `field:assureur`).

    Args:
        field: Schema field name.

    Returns:
        The component key.
    """
    return f"{FIELD_PREFIX}{field}"


def seed_candidate() -> Candidate:
    """Build the initial candidate from the schema's field descriptions.

    Returns:
        A candidate with `instructions` and one `field:<name>` per schema field.
    """
    candidate = {INSTRUCTIONS_KEY: SEED_INSTRUCTIONS}
    for name, description in field_descriptions().items():
        candidate[component_key(name)] = description
    return candidate


def validate_candidate(candidate: Candidate) -> None:
    """Check that a candidate has exactly the expected components.

    Args:
        candidate: Candidate to check.

    Raises:
        ValueError: If a component is missing, unexpected or empty.
    """
    expected = {INSTRUCTIONS_KEY, *(component_key(name) for name in FIELD_NAMES)}
    missing = expected - candidate.keys()
    extra = candidate.keys() - expected
    empty = [key for key in expected & candidate.keys() if not candidate[key].strip()]
    if missing or extra or empty:
        raise ValueError(f"Invalid candidate: missing={sorted(missing)} extra={sorted(extra)} empty={sorted(empty)}")


def render(candidate: Candidate) -> RenderedPrompt:
    """Render a candidate into the final system prompt and JSON schema.

    Args:
        candidate: A valid candidate.

    Returns:
        The rendered prompt.

    Raises:
        ValueError: If the candidate is invalid.
    """
    validate_candidate(candidate)
    descriptions = {name: candidate[component_key(name)] for name in FIELD_NAMES}
    field_lines = "\n".join(f"- {name} : {text}" for name, text in descriptions.items())
    system = f"{candidate[INSTRUCTIONS_KEY]}\n\nChamps à extraire :\n{field_lines}"
    return RenderedPrompt(system=system, json_schema=strict_json_schema(descriptions))


def candidate_hash(candidate: Candidate) -> str:
    """Return the sha256 of the canonical JSON form of a candidate.

    Args:
        candidate: Candidate to hash.

    Returns:
        Hex digest.
    """
    canonical = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_candidate(candidate: Candidate, prompts_dir: Path, name: str) -> Path:
    """Save a candidate as versioned JSON.

    Saving the same content twice under the same name rewrites the same file.

    Args:
        candidate: Candidate to save.
        prompts_dir: Destination directory (`paths.prompts`).
        name: Human-readable prefix, e.g. `seed` or `gepa-best`.

    Returns:
        Path of the written file, `<prompts_dir>/<name>.<hash8>.json`.
    """
    validate_candidate(candidate)
    digest = candidate_hash(candidate)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / f"{name}.{digest[:8]}.json"
    payload = {
        "name": name,
        "hash": digest,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "candidate": candidate,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_candidate(path: Path) -> Candidate:
    """Load a saved candidate and check its hash.

    Args:
        path: File written by `save_candidate`.

    Returns:
        The candidate.

    Raises:
        ValueError: If the stored hash does not match the content, or the
            candidate is invalid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate: Candidate = payload["candidate"]
    if candidate_hash(candidate) != payload["hash"]:
        raise ValueError(f"{path}: content does not match its stored hash (edited by hand?)")
    validate_candidate(candidate)
    return candidate

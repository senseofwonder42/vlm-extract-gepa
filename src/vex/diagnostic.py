"""Diagnose one extraction error by looking at the document.

For a failing field, the `diagnostic` model receives the page images, the
extracted value and the gold value, and must answer:

    {"located": bool, "quote": str, "location": str, "explanation": str}

- `located=true`: it found the gold value in the document. `quote` is the
  exact text around it and `location` says where (box, header, line...).
  `explanation` says why the extraction went wrong. This is what GEPA's
  reflection model will read.
- `located=false`: it could not find the gold value. Either the model
  missed it or the ANNOTATION may be wrong. Such cases are written to
  `data/review_queue/suspect_annotations.jsonl` for a human to check and are
  kept out of reflection, so we never teach the prompt a possible label error.

Quoting is mandatory: a diagnosis without a quote or a location is dropped.
Diagnoses are cached on disk by (doc_id, field, hash(pred), hash(gold)), plus
the diagnostic model id and the prompt version, so swapping the model in YAML
never serves stale verdicts.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, JsonValue

from vex.annotation.store import now_iso
from vex.extract import build_messages
from vex.llm.client import OpenRouterClient

PROMPT_VERSION = "diag-v1"

DIAGNOSTIC_PROMPT = """Tu vérifies une extraction de champ sur un justificatif d'assurance automobile français.

Champ : {field}
Description du champ : {description}
Valeur extraite par le modèle : {pred}
Valeur attendue (annotation humaine) : {gold}

Cherche la valeur attendue dans le document (images ci-dessous).
- Si tu la trouves : located=true, quote = le texte exact du document autour de la valeur
  (obligatoire), location = où elle se trouve (case, bloc, ligne, page), explanation = pourquoi
  l'extraction s'est trompée (mauvaise zone, confusion avec un autre champ, format, valeur absente…).
- Si la valeur attendue est null : vérifie que le champ est bien absent ; located=true si tu
  confirmes son absence (quote = la zone où il aurait dû être), false si tu vois en fait une valeur.
- Si tu ne trouves pas la valeur attendue : located=false et explique ce que tu vois à la place.
Réponds uniquement en JSON."""

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "located": {"type": "boolean"},
        "quote": {"type": "string"},
        "location": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["located", "quote", "location", "explanation"],
    "additionalProperties": False,
}


class Diagnosis(BaseModel):
    """The diagnostic model's verdict on one failing field."""

    located: bool
    quote: str
    location: str
    explanation: str


@dataclass(frozen=True)
class FieldDiagnosis:
    """A usable diagnosis (located, with quote and location).

    Attributes:
        doc_id: Document.
        field: Field name.
        quote: Exact text around the gold value.
        location: Where it is.
        explanation: Why the extraction failed.
    """

    doc_id: str
    field: str
    quote: str
    location: str
    explanation: str


def _sha(value: JsonValue) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class Diagnostician:
    """Runs, caches and filters diagnoses."""

    def __init__(self, client: OpenRouterClient, cache_dir: Path, review_queue: Path) -> None:
        """Create the diagnostician.

        Args:
            client: OpenRouter client (uses the `diagnostic` role).
            cache_dir: Directory of cached diagnoses.
            review_queue: Directory holding `suspect_annotations.jsonl`.
        """
        self.client = client
        self.cache_dir = cache_dir
        self.suspects_path = review_queue / "suspect_annotations.jsonl"

    def _cache_path(self, doc_id: str, field: str, pred: JsonValue, gold: JsonValue) -> Path:
        model = self.client.config.role("diagnostic").model
        key = hashlib.sha256(
            f"{doc_id}|{field}|{_sha(pred)}|{_sha(gold)}|{model}|{PROMPT_VERSION}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    async def _ask(self, document: Path, field: str, description: str, pred: JsonValue, gold: JsonValue) -> Diagnosis | None:
        prompt = DIAGNOSTIC_PROMPT.format(field=field, description=description,
                                          pred=json.dumps(pred, ensure_ascii=False),
                                          gold=json.dumps(gold, ensure_ascii=False))
        render = self.client.config.render
        messages = build_messages(prompt, document, render.dpi, render.max_pages)
        result = await self.client.complete_json("diagnostic", messages, Diagnosis, DIAGNOSIS_SCHEMA,
                                                 schema_name="diagnosis")
        return result.parsed

    async def diagnose(
        self, doc_id: str, document: Path, field: str, description: str, pred: JsonValue, gold: JsonValue
    ) -> FieldDiagnosis | None:
        """Diagnose one failing field.

        Args:
            doc_id: Document id.
            document: Document file.
            field: Failing field.
            description: Current description of the field (the candidate's).
            pred: Extracted value.
            gold: Gold value.

        Returns:
            A usable diagnosis, or `None` if it was dropped (no quote/location,
            unparseable answer) or flagged as a suspect annotation.
        """
        path = self._cache_path(doc_id, field, pred, gold)
        if path.exists():
            diagnosis: Diagnosis | None = Diagnosis.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            diagnosis = await self._ask(document, field, description, pred, gold)
            if diagnosis is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(diagnosis.model_dump_json(), encoding="utf-8")
        if diagnosis is None:
            return None
        if not diagnosis.located:
            self._flag_suspect(doc_id, field, pred, gold, diagnosis)
            return None
        if not diagnosis.quote.strip() or not diagnosis.location.strip():
            logger.debug("{} {}: diagnosis without quote/location dropped", doc_id[:12], field)
            return None
        return FieldDiagnosis(doc_id, field, diagnosis.quote.strip(), diagnosis.location.strip(),
                              diagnosis.explanation.strip())

    def _flag_suspect(self, doc_id: str, field: str, pred: JsonValue, gold: JsonValue, diagnosis: Diagnosis) -> None:
        entry = {"doc_id": doc_id, "field": field, "pred": pred, "gold": gold,
                 "diagnostic_explanation": diagnosis.explanation, "diagnostic_model":
                 self.client.config.role("diagnostic").model, "flagged_at": now_iso()}
        existing = set()
        if self.suspects_path.exists():
            for line in self.suspects_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                existing.add((item["doc_id"], item["field"], _sha(item["gold"])))
        if (doc_id, field, _sha(gold)) in existing:
            return
        self.suspects_path.parent.mkdir(parents=True, exist_ok=True)
        with self.suspects_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.warning("Suspect annotation: {} {} (gold {!r} not found in the document)", doc_id[:12], field, gold)

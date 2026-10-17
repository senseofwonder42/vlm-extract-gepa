"""LocalBackend: one editable JSON file per document, so the loop runs on a laptop.

`push` writes `data/inbox/round_<N>/<doc_id>.json`:

    {
      "_instructions": "...",
      "_done": false,                <- set to true when finished
      "doc_id": "...", "document": "data/raw/...", "status": "pre_annotated",
      "values": {...},               <- prefilled with the model's values (all null if blind)
      "comment": "",                 <- free text: ambiguities, doubts
      "annotator": "",               <- your name
      "review_seconds": null         <- optional: how long the review took
    }

`pull` imports every file marked `"_done": true`, then renames it to
`*.imported.json` so it is never imported twice. A file whose values do not
fit the schema (e.g. an unparseable date) is reported and left in place.
"""

import json
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from vex.annotation.backend import PushItem, clean_human_values
from vex.annotation.store import AnnotationRecord, now_iso
from vex.schema import FIELD_NAMES

INSTRUCTIONS = (
    "Open `document`, correct `values` so they match the document (null if a field is absent "
    "or unreadable, never an empty string), dates as YYYY-MM-DD or JJ/MM/AAAA, garanties as a "
    "list. Write any doubt in `comment`, your name in `annotator`, then set `_done` to true."
)


class LocalBackend:
    """File-based annotation backend."""

    def __init__(self, inbox: Path) -> None:
        """Create the backend.

        Args:
            inbox: `data/inbox` (git-ignored).
        """
        self.inbox = inbox

    def _round_dir(self, round_number: int) -> Path:
        return self.inbox / f"round_{round_number}"

    def push(self, items: list[PushItem], round_number: int) -> None:
        """Write one editable file per document.

        Args:
            items: Documents to annotate.
            round_number: Round number.
        """
        folder = self._round_dir(round_number)
        folder.mkdir(parents=True, exist_ok=True)
        for item in items:
            payload = {
                "_instructions": INSTRUCTIONS,
                "_done": False,
                "doc_id": item.doc_id,
                "document": str(item.path),
                "status": item.status,
                "values": item.values if item.values is not None else dict.fromkeys(FIELD_NAMES),
                "comment": "",
                "annotator": "",
                "review_seconds": None,
            }
            path = folder / f"{item.doc_id}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("Wrote {} file(s) to {}: edit them, set _done to true, then pull", len(items), folder)

    def pull(self, round_number: int) -> list[AnnotationRecord]:
        """Import finished files of a round.

        Args:
            round_number: Round number.

        Returns:
            One `reviewed` or `blind` record per finished, valid file.
        """
        records = []
        for path in sorted(self._round_dir(round_number).glob("*.json")):
            if path.name.endswith(".imported.json"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("_done"):
                continue
            try:
                values = clean_human_values(payload["values"])
            except ValidationError as exc:
                logger.error("{}: not imported, fix the values ({})", path.name, exc.errors()[0]["msg"])
                continue
            records.append(AnnotationRecord(
                doc_id=payload["doc_id"],
                round=round_number,
                status="blind" if payload["status"] == "blind" else "reviewed",
                values=values,
                source="human",
                annotator=f"local:{payload.get('annotator') or 'unknown'}",
                review_seconds=payload.get("review_seconds"),
                comment=payload.get("comment") or None,
                backend="local",
                reviewed_at=now_iso(),
            ))
            path.rename(path.with_name(path.stem + ".imported.json"))
        logger.info("Imported {} finished file(s) from round {}", len(records), round_number)
        return records

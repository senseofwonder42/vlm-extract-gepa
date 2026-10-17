"""KiliBackend: drive a Kili project (SDK 2.142.1) for review rounds.

UNTESTED AGAINST A LIVE KILI INSTANCE. Kili is not reachable from the
development machine. The SDK signatures used here were read from the
installed 2.142.1 source, the payload shapes are validated offline with the
SDK's own parser (tests/test_kili_payloads.py), and the patterns follow
~/Projects/template-kili (itself never run against a server). Check
docs/kili_uncertainties.md before the first run at work.

Project: input type PDF. PNG/JPEG documents are wrapped into a one-page PDF
before upload so a single project holds every document. Each asset's
`external_id` is the doc_id; its metadata carries status and round.
Pre-annotations are uploaded with `create_predictions` (never for blind items).
"""

import tempfile
from pathlib import Path

import pymupdf
from loguru import logger
from pydantic import ValidationError

from vex.annotation.backend import PushItem
from vex.annotation.kili_payloads import (
    METADATA_KEYS,
    asset_metadata,
    json_interface,
    label_to_record,
    latest_labels,
    prediction_response,
)
from vex.annotation.store import AnnotationRecord
from vex.config import Config, Settings

LABEL_FIELDS = ["id", "jsonResponse", "labelType", "createdAt", "secondsToLabel", "author.email", "labelOf.externalId"]


class MissingKiliCredentialsError(RuntimeError):
    """KILI_API_KEY or KILI_API_ENDPOINT is missing from `.env`."""


def as_pdf(path: Path, workdir: Path) -> Path:
    """Return a PDF version of a document (images are wrapped in a one-page PDF).

    Args:
        path: PDF, PNG or JPEG file.
        workdir: Where to write converted files.

    Returns:
        `path` itself for PDFs, else the converted file.
    """
    if path.suffix.lower() == ".pdf":
        return path
    target = workdir / f"{path.stem}.pdf"
    with pymupdf.open(path) as image, pymupdf.open("pdf", image.convert_to_pdf()) as pdf:
        pdf.save(target)
    return target


class KiliBackend:
    """Kili-based annotation backend (see module docstring: untested live)."""

    def __init__(self, config: Config, settings: Settings) -> None:
        """Connect to Kili.

        Args:
            config: Resolved configuration (`annotation.kili`).
            settings: Secrets; `KILI_API_KEY` and `KILI_API_ENDPOINT` are required.

        Raises:
            MissingKiliCredentialsError: If a credential is missing (fail early,
                as template-kili's `get_kili` does).
        """
        if settings.kili_api_key is None or not settings.kili_api_endpoint:
            raise MissingKiliCredentialsError("Set KILI_API_KEY and KILI_API_ENDPOINT in .env (see .env.example)")
        from kili.client import Kili  # imported here: never needed with the local backend

        self.config = config.annotation.kili
        # api_endpoint is mandatory on-premise, otherwise the SDK targets Kili SaaS.
        self.kili = Kili(api_key=settings.kili_api_key.get_secret_value(), api_endpoint=settings.kili_api_endpoint)
        self.project_id = self.config.project_id or self._create_project()

    def _create_project(self) -> str:
        project = self.kili.create_project(
            title=self.config.project_title,
            description="Relecture des attestations d'assurance automobile (vlm-extract-gepa).",
            input_type="PDF",
            json_interface=json_interface(),
        )
        project_id = project["id"]
        logger.warning("Created Kili project {}: set annotation.kili.project_id to it in your config", project_id)
        return project_id

    def push(self, items: list[PushItem], round_number: int) -> None:
        """Upload assets with metadata, then predictions for non-blind items.

        Args:
            items: Documents to annotate.
            round_number: Round number.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdfs = [str(as_pdf(item.path, Path(tmp))) for item in items]
            self.kili.append_many_to_dataset(
                project_id=self.project_id,
                content_array=pdfs,
                external_id_array=[item.doc_id for item in items],
                json_metadata_array=[asset_metadata(i.doc_id, i.origin, i.status, round_number) for i in items],
            )
        to_predict = [item for item in items if item.status == "pre_annotated" and item.values is not None]
        if to_predict:
            self.kili.create_predictions(
                project_id=self.project_id,
                external_id_array=[item.doc_id for item in to_predict],
                json_response_array=[prediction_response(item.values or {}) for item in to_predict],
                model_name=f"{self.config.model_name_prefix}-round{round_number}",
            )
        logger.info("Pushed {} asset(s) to Kili ({} with predictions)", len(items), len(to_predict))

    def pull(self, round_number: int) -> list[AnnotationRecord]:
        """Export human labels of a round's assets and convert them to records.

        Args:
            round_number: Round number.

        Returns:
            One record per labelled asset of the round.
        """
        assets = list(self.kili.assets(
            project_id=self.project_id,
            fields=["externalId", "jsonMetadata"],
            metadata_where={"round": round_number},
        ))
        status_by_id = {
            a["externalId"]: ("blind" if (a.get("jsonMetadata") or {}).get("status") == "blind" else "reviewed")
            for a in assets
            if set(METADATA_KEYS) <= set(a.get("jsonMetadata") or {})
        }
        labels = list(self.kili.labels(
            project_id=self.project_id,
            asset_external_id_in=list(status_by_id),
            type_in=["DEFAULT", "REVIEW"],
            fields=LABEL_FIELDS,
        ))
        records = []
        for external_id, label in latest_labels(labels).items():
            try:
                records.append(label_to_record(label, status_by_id[external_id], round_number))
            except ValidationError as exc:
                logger.error("Kili label for {} not imported: {}", external_id, exc.errors()[0]["msg"])
        logger.info("Pulled {} labelled asset(s) of round {} from Kili", len(records), round_number)
        return records

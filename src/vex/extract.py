"""Run one extraction: a document file + a candidate prompt → `ExtractionResult`.

This is the "program" GEPA optimizes. A failed parse (after the repair retry)
is not an exception: it comes back with `values=None`, and the metric types
every field as an omission.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from vex.config import RoleName
from vex.llm.client import OpenRouterClient, StructuredPath
from vex.llm.media import load_pages
from vex.prompt import Candidate, candidate_hash, render
from vex.schema import AttestationAuto


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of extracting one document.

    Attributes:
        document: Path of the document file.
        role: Role used (`extractor_eval`, `extractor`, `preannotator`).
        model: Model id.
        candidate_hash: Hash of the candidate prompt.
        values: Extracted fields (`AttestationAuto` dump), `None` if parsing failed.
        raw_outputs: Raw assistant text of each attempt.
        structured_path: `json_schema` or `prompt`.
        repaired: True if the repair retry was used.
        error: Last parse error when `values` is `None`.
        usd: Cost of this extraction (0 when fully cached).
    """

    document: str
    role: str
    model: str
    candidate_hash: str
    values: dict[str, JsonValue] | None
    raw_outputs: list[str]
    structured_path: StructuredPath
    repaired: bool
    error: str | None
    usd: float

    @property
    def failed(self) -> bool:
        """True if no valid JSON was obtained."""
        return self.values is None


def build_messages(system: str, document: Path, dpi: int, max_pages: int) -> list[dict]:
    """Build the chat messages for one document.

    Args:
        system: Rendered system prompt.
        document: Document file (PDF, PNG, JPEG).
        dpi: PDF rendering resolution.
        max_pages: Maximum number of pages sent.

    Returns:
        A system message and a user message holding every page image.
    """
    pages = load_pages(document, dpi=dpi, max_pages=max_pages)
    content: list[dict] = [{"type": "text", "text": f"Document ({len(pages)} page(s)) :"}]
    content.extend(page.content_part() for page in pages)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


async def extract_document(
    client: OpenRouterClient,
    document: Path,
    candidate: Candidate,
    role: RoleName = "extractor_eval",
    *,
    use_cache: bool = True,
    cache_salt: str | None = None,
) -> ExtractionResult:
    """Extract the fields of one document with a candidate prompt.

    Args:
        client: OpenRouter client.
        document: Document file.
        candidate: Candidate prompt.
        role: Role to use.
        use_cache: Per-call cache switch.
        cache_salt: Extra cache-key component (self-consistency sample index).

    Returns:
        The extraction result; never raises on invalid model output.
    """
    rendered = render(candidate)
    render_config = client.config.render
    messages = build_messages(rendered.system, document, render_config.dpi, render_config.max_pages)
    result = await client.complete_json(
        role, messages, AttestationAuto, rendered.json_schema,
        schema_name="attestation_auto", use_cache=use_cache, cache_salt=cache_salt,
    )
    return ExtractionResult(
        document=str(document),
        role=role,
        model=client.config.role(role).model,
        candidate_hash=candidate_hash(candidate),
        values=result.parsed.model_dump(mode="json") if result.parsed else None,
        raw_outputs=result.raw_outputs,
        structured_path=result.path,
        repaired=result.repaired,
        error=result.error,
        usd=sum(usage.usd for usage in result.usages),
    )

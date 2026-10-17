"""Async OpenRouter client shared by every role.

The request body is built as::

    {"model": role.model, "messages": ..., **role.params, **role.extra_body,
     "response_format": ...}   # response_format only on the json_schema path

`params` and `extra_body` are passed through unchanged, so any OpenRouter or
provider parameter (top_k, min_p, reasoning, provider routing, ...) works from
YAML without code changes. httpx is used directly rather than the `openai` SDK
because the SDK rejects non-OpenAI keyword arguments.

Structured output has two paths, logged on every call:

- `json_schema`: strict `response_format` built from the Pydantic schema, used
  when the routed endpoints advertise `structured_outputs`;
- `prompt`: the schema is appended to the system prompt and the reply is parsed.

On both paths, an unparseable or invalid reply gets ONE repair retry (the reply
and the validation error are sent back). If that also fails, the result carries
`parsed=None` and the caller decides how to score it.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Literal

import httpx
from loguru import logger
from pydantic import BaseModel, SecretStr, ValidationError

from vex.config import Config, RoleName
from vex.llm.cache import DiskCache, cache_key
from vex.llm.cost import CallUsage, CostTracker, usage_from_response

StructuredPath = Literal["json_schema", "prompt"]

REPAIR_PROMPT = (
    "Ta réponse précédente n'est pas un JSON valide pour le schéma demandé.\n"
    "Erreur : {error}\n"
    "Renvoie uniquement l'objet JSON corrigé, sans texte autour."
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMError(RuntimeError):
    """A call failed for good: non-retryable error, or retries exhausted."""


@dataclass(frozen=True)
class Completion:
    """One chat completion.

    Attributes:
        content: Assistant message text (empty string if the provider sent none).
        usage: Tokens and cost of the call.
        response: Full response JSON, as cached.
    """

    content: str
    usage: CallUsage
    response: dict


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    """Outcome of a structured (JSON) call.

    Attributes:
        parsed: Validated model, or `None` if both attempts failed.
        raw_outputs: Assistant text of each attempt (1 or 2 items).
        path: Structured-output path used.
        repaired: True if the repair retry was needed.
        error: Last parse or validation error, `None` on success.
        usages: Usage of each attempt.
    """

    parsed: T | None
    raw_outputs: list[str]
    path: StructuredPath
    repaired: bool
    error: str | None
    usages: list[CallUsage]


def _is_retryable(code: int) -> bool:
    return code in (408, 429) or code >= 500


def parse_json_reply[T: BaseModel](text: str, model_cls: type[T]) -> tuple[T | None, str | None]:
    """Parse and validate a JSON reply, tolerating code fences and surrounding text.

    Args:
        text: Assistant reply.
        model_cls: Pydantic model to validate against.

    Returns:
        `(model, None)` on success, `(None, error message)` otherwise.
    """
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        return None, "no JSON object found in the reply"
    try:
        return model_cls.model_validate(json.loads(cleaned[start : end + 1])), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except ValidationError as exc:
        return None, f"schema validation failed: {exc}"[:1500]


def _with_schema_instruction(messages: list[dict], json_schema: dict) -> list[dict]:
    instruction = "Schéma JSON attendu :\n" + json.dumps(json_schema, ensure_ascii=False)
    messages = [dict(message) for message in messages]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


class OpenRouterClient:
    """Async client: retries, concurrency limit, disk cache, cost tracking."""

    def __init__(
        self,
        config: Config,
        api_key: SecretStr,
        tracker: CostTracker,
        *,
        use_cache: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create the client.

        Args:
            config: Resolved configuration.
            api_key: OpenRouter key (from `Settings`).
            tracker: Run cost tracker; also enforces the budget.
            use_cache: Run-level cache switch (`--no-cache` sets it to False).
                Ignored when `cache.enabled` is false.
            transport: Optional httpx transport, used by tests to mock the network.
        """
        self.config = config
        self.tracker = tracker
        self._cache = DiskCache(config.cache.dir) if (config.cache.enabled and use_cache) else None
        self._semaphore = asyncio.Semaphore(config.client.concurrency)
        self._paths: dict[RoleName, StructuredPath] = {}
        self._http = httpx.AsyncClient(
            base_url=config.client.base_url,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=config.client.timeout_s,
            transport=transport,
        )

    async def __aenter__(self) -> "OpenRouterClient":
        """Enter the async context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the HTTP connection pool."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._http.aclose()

    async def complete(
        self,
        role: RoleName,
        messages: list[dict],
        *,
        response_format: dict | None = None,
        use_cache: bool = True,
        cache_salt: str | None = None,
    ) -> Completion:
        """Send one chat completion for a role.

        Args:
            role: Role whose model, params and extra_body are used.
            messages: OpenAI-style messages (content may hold image parts).
            response_format: Optional `response_format` value.
            use_cache: Per-call cache switch.
            cache_salt: Extra cache-key component, e.g. a sample index for
                self-consistency passes, so repeated samples are not replayed.

        Returns:
            The completion.

        Raises:
            BudgetExceeded: If the run budget is spent (checked before sending).
            LLMError: On a non-retryable error or once retries are exhausted.
        """
        role_config = self.config.role(role)
        body: dict = {"model": role_config.model, "messages": messages, **role_config.params, **role_config.extra_body}
        if response_format is not None:
            body["response_format"] = response_format
        key = cache_key({**body, "_salt": cache_salt})
        cache = self._cache if use_cache else None

        cached_response = cache.get(key) if cache else None
        if cached_response is not None:
            response, cached = cached_response, True
        else:
            self.tracker.check_budget()
            response, cached = await self._post(body), False
            if cache:
                cache.set(key, response)

        usage = usage_from_response(
            response, role, role_config.model, self.config.pricing_fallback_usd_per_m, cached
        )
        self.tracker.record(usage)
        logger.debug(
            "{} {} cached={} in={} out={} reasoning={} usd={:.6f}",
            role,
            role_config.model,
            cached,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.reasoning_tokens,
            usage.usd,
        )
        content = response["choices"][0]["message"].get("content") or ""
        return Completion(content=content, usage=usage, response=response)

    async def _post(self, body: dict) -> dict:
        settings = self.config.client
        async with self._semaphore:
            for attempt in range(settings.max_retries + 1):
                retry_after: float | None = None
                try:
                    response = await self._http.post("/chat/completions", json=body)
                except httpx.TransportError as exc:
                    reason = f"transport error: {exc!r}"
                else:
                    try:
                        data = response.json()
                    except ValueError:
                        data = {}
                    error = data.get("error") if isinstance(data, dict) else None
                    if response.status_code == 200 and not error and data.get("choices"):
                        return data
                    code = response.status_code if response.status_code != 200 else int((error or {}).get("code") or 502)
                    message = str((error or {}).get("message") or response.text)[:500]
                    if not _is_retryable(code):
                        raise LLMError(f"OpenRouter error {code} for {body['model']}: {message}")
                    reason = f"HTTP {code}: {message}"
                    header = response.headers.get("retry-after")
                    retry_after = float(header) if header and header.replace(".", "", 1).isdigit() else None
                if attempt == settings.max_retries:
                    raise LLMError(f"Giving up on {body['model']} after {attempt + 1} attempts ({reason})")
                delay = min(settings.backoff_max_s, settings.backoff_base_s * 2**attempt)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                logger.warning("{} ({}), retry {}/{} in {:.1f}s", body["model"], reason, attempt + 1, settings.max_retries, delay)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def structured_path(self, role: RoleName) -> StructuredPath:
        """Decide whether a role uses strict `response_format` or prompt-only JSON.

        In `auto` mode, the model's endpoint list is fetched once per client and
        the pinned providers (`extra_body.provider.order`, or all endpoints when
        nothing is pinned) are checked for `structured_outputs` support.

        Args:
            role: Role name.

        Returns:
            `json_schema` or `prompt`.
        """
        if role in self._paths:
            return self._paths[role]
        role_config = self.config.role(role)
        path: StructuredPath
        if role_config.structured_output != "auto":
            path = role_config.structured_output
        else:
            path = await self._probe_structured_support(role_config.model, role_config.extra_body.get("provider"))
        logger.info("Structured output for role {} ({}): {}", role, role_config.model, path)
        self._paths[role] = path
        return path

    async def _probe_structured_support(self, model: str, provider: object) -> StructuredPath:
        try:
            response = await self._http.get(f"/models/{model}/endpoints")
            response.raise_for_status()
            endpoints = response.json()["data"]["endpoints"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("Could not read endpoints of {} ({!r}); falling back to prompt-only JSON", model, exc)
            return "prompt"
        order = provider.get("order") if isinstance(provider, dict) else None
        if order:
            slugs = {str(slug).lower() for slug in order}
            endpoints = [e for e in endpoints if str(e.get("tag", "")).split("/")[0].lower() in slugs]
        supported = any("structured_outputs" in e.get("supported_parameters", []) for e in endpoints)
        return "json_schema" if supported else "prompt"

    async def complete_json[T: BaseModel](
        self,
        role: RoleName,
        messages: list[dict],
        model_cls: type[T],
        json_schema: dict,
        *,
        schema_name: str = "extraction",
        use_cache: bool = True,
        cache_salt: str | None = None,
    ) -> StructuredResult[T]:
        """Get a reply validated against `model_cls`, with one repair retry.

        Args:
            role: Role name.
            messages: Conversation; its system message gets the schema appended
                on the `prompt` path.
            model_cls: Pydantic model the reply must validate against.
            json_schema: Strict JSON schema matching `model_cls`.
            schema_name: Name given to the schema in `response_format`.
            use_cache: Per-call cache switch.
            cache_salt: See `complete`.

        Returns:
            The structured result; `parsed` is `None` if both attempts failed.
        """
        path = await self.structured_path(role)
        if path == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
            }
        else:
            response_format = None
            messages = _with_schema_instruction(messages, json_schema)

        first = await self.complete(
            role, messages, response_format=response_format, use_cache=use_cache, cache_salt=cache_salt
        )
        parsed, error = parse_json_reply(first.content, model_cls)
        if parsed is not None:
            return StructuredResult(parsed, [first.content], path, repaired=False, error=None, usages=[first.usage])

        logger.warning("{}: invalid reply on path {} ({}); one repair retry", role, path, (error or "")[:200])
        repair_messages = [
            *messages,
            {"role": "assistant", "content": first.content},
            {"role": "user", "content": REPAIR_PROMPT.format(error=error)},
        ]
        second = await self.complete(
            role, repair_messages, response_format=response_format, use_cache=use_cache, cache_salt=cache_salt
        )
        parsed, error = parse_json_reply(second.content, model_cls)
        if parsed is None:
            logger.error("{}: reply still invalid after repair ({})", role, (error or "")[:200])
        return StructuredResult(
            parsed, [first.content, second.content], path, repaired=True, error=error,
            usages=[first.usage, second.usage],
        )

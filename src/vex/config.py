"""Configuration: YAML validated into Pydantic models, secrets from `.env`.

The YAML file (`configs/default.yaml`) holds every tunable. Any key can be
overridden from the CLI with `--set dotted.key=value`, where `value` is parsed
as YAML (so `0.2`, `true`, `[a, b]` and `{k: v}` all work). Secrets live only in
`.env` and are loaded by `Settings`; they never enter `Config`, so they are never
written to a run directory or included in the config hash.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, JsonValue, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vex.schema import FIELD_NAMES

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

RoleName = Literal["extractor", "extractor_eval", "preannotator", "diagnostic", "reflection"]
StructuredOutputMode = Literal["auto", "json_schema", "prompt"]


class _Strict(BaseModel):
    """Base model rejecting unknown keys, so a typo in YAML or `--set` fails loudly."""

    model_config = ConfigDict(extra="forbid")


class RoleConfig(_Strict):
    """One model role. `params` and `extra_body` are sent to OpenRouter unchanged.

    Attributes:
        model: OpenRouter model id.
        structured_output: `json_schema` forces `response_format`, `prompt` asks for
            JSON in the prompt only, `auto` uses `json_schema` when the routed
            endpoints advertise `structured_outputs`.
        params: Sampling and length parameters (temperature, top_k, max_tokens, ...).
        extra_body: Any other request-body keys (reasoning, provider routing, ...).
    """

    model: str
    structured_output: StructuredOutputMode = "auto"
    params: dict[str, JsonValue] = {}
    extra_body: dict[str, JsonValue] = {}


class RoleOverride(_Strict):
    """Partial role config deep-merged over another role (used by `extractor_eval`)."""

    params: dict[str, JsonValue] = {}
    extra_body: dict[str, JsonValue] = {}


class ModelsConfig(_Strict):
    """All model roles."""

    extractor: RoleConfig
    extractor_eval: RoleOverride
    preannotator: RoleConfig
    diagnostic: RoleConfig
    reflection: RoleConfig


class Price(_Strict):
    """Price in USD per million tokens."""

    input: float
    output: float


class RenderConfig(_Strict):
    """PDF rendering."""

    dpi: int = 150
    max_pages: int = 3


class ClientConfig(_Strict):
    """HTTP client behaviour."""

    base_url: str = "https://openrouter.ai/api/v1"
    concurrency: int = 8
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    timeout_s: float = 120.0


class CacheConfig(_Strict):
    """Disk cache for LLM responses."""

    enabled: bool = True
    dir: Path = Path(".cache/llm")


class BudgetConfig(_Strict):
    """Spending limit for one run.

    Attributes:
        max_usd: Hard limit for the whole run.
        heldout_reserve_usd: Part of `max_usd` kept for held-out scoring after GEPA.
    """

    max_usd: float
    heldout_reserve_usd: float = 0.5


class PathsConfig(_Strict):
    """Repository paths (relative to the working directory)."""

    raw: Path
    registry: Path
    manifest: Path
    specimen_sources: Path
    synthetic_meta: Path
    splits: Path
    annotations: Path
    inbox: Path
    prompts: Path
    runs: Path
    review_queue: Path
    reports: Path


class SynthConfig(_Strict):
    """Synthetic certificate generation."""

    n: int
    seed: int
    scan_fraction: float


class SplitsConfig(_Strict):
    """Split assignment parameters."""

    seed: int
    heldout_blind_size: int
    pareto_fraction: float
    standin_layouts: list[str] = []
    heldout_standin_size: int = 15


class MetricConfig(_Strict):
    """Field weights and comparison settings."""

    weights: dict[str, float]
    fuzzy_fields: list[str]
    fuzzy_threshold: float

    @model_validator(mode="after")
    def _fields_match_schema(self) -> "MetricConfig":
        """Require one positive weight per schema field and known fuzzy fields."""
        if set(self.weights) != set(FIELD_NAMES):
            raise ValueError(f"metric.weights must list exactly the schema fields {list(FIELD_NAMES)}")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("metric.weights must be positive")
        if unknown := set(self.fuzzy_fields) - set(FIELD_NAMES):
            raise ValueError(f"metric.fuzzy_fields has unknown fields: {sorted(unknown)}")
        return self


class KiliConfig(_Strict):
    """Kili project settings (used only with `annotation.backend: kili`)."""

    project_id: str | None = None
    project_title: str
    model_name_prefix: str


class AnnotationConfig(_Strict):
    """Annotation backend selection."""

    backend: Literal["local", "kili"]
    kili: KiliConfig


class BootstrapConfig(_Strict):
    """Bootstrap micro-batch sizes, one per round."""

    batch_sizes: list[int]


class GepaConfig(_Strict):
    """Arguments forwarded to `gepa.optimize`, plus held-out scoring settings."""

    seed: int
    max_metric_calls: int
    max_reflection_cost_usd: float
    reflection_minibatch_size: int
    candidate_selection_strategy: Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"]
    module_selector: Literal["error_driven", "round_robin", "all"]
    use_merge: bool
    max_merge_invocations: int
    skip_perfect_score: bool
    cache_evaluation: bool
    heldout_top_k: int
    reflection_images: bool


class SelfConsistencyConfig(_Strict):
    """Self-consistency sampling."""

    n: int
    temperature: float


class DiversityConfig(_Strict):
    """Diversity sampling via image-embedding clusters."""

    enabled: bool
    n_clusters: int
    embedder: Literal["auto", "onnx", "thumbnail"]


class ActiveConfig(_Strict):
    """Active-learning selection."""

    weights: dict[str, float]
    top_k_candidates: int
    self_consistency: SelfConsistencyConfig
    verifier_shortlist: int
    diversity: DiversityConfig
    random_confident_fraction: float
    batch_size: int
    max_pool: int = 60


class TrackingConfig(_Strict):
    """Per-round tracking and plateau detection."""

    plateau_delta: float
    plateau_rounds: int


class Config(_Strict):
    """The full, validated configuration."""

    models: ModelsConfig
    pricing_fallback_usd_per_m: dict[str, Price]
    render: RenderConfig
    client: ClientConfig
    cache: CacheConfig
    budget: BudgetConfig
    paths: PathsConfig
    synth: SynthConfig
    splits: SplitsConfig
    metric: MetricConfig
    annotation: AnnotationConfig
    bootstrap: BootstrapConfig
    gepa: GepaConfig
    active: ActiveConfig
    tracking: TrackingConfig

    def role(self, name: RoleName) -> RoleConfig:
        """Return the effective config of a role.

        `extractor_eval` is `extractor` with its overrides deep-merged on top.

        Args:
            name: Role name.

        Returns:
            The resolved role configuration.
        """
        if name == "extractor_eval":
            base = self.models.extractor
            override = self.models.extractor_eval
            return base.model_copy(
                update={
                    "params": _deep_merge(base.params, override.params),
                    "extra_body": _deep_merge(base.extra_body, override.extra_body),
                }
            )
        return getattr(self.models, name)


class Settings(BaseSettings):
    """Secrets read from `.env` (or the environment). Never logged or persisted."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: SecretStr | None = None
    kili_api_key: SecretStr | None = None
    kili_api_endpoint: str | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_override(data: dict, assignment: str) -> None:
    """Apply one `dotted.key=value` override in place.

    Intermediate keys are created when missing, so new entries can be added to
    free-form dicts such as `params`. Typos in typed sections are caught later,
    when the result is validated.

    Args:
        data: Raw config dict (as loaded from YAML).
        assignment: Override such as `models.extractor.params.temperature=0.2`.

    Raises:
        ValueError: If the assignment has no `=` or an empty key, or if it
            descends into a non-mapping value.
    """
    key, sep, raw_value = assignment.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Override must look like 'a.b.c=value', got {assignment!r}")
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {key!r}: {part!r} is not a mapping")
        node = child
    node[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path: Path = DEFAULT_CONFIG_PATH, overrides: list[str] | None = None) -> Config:
    """Load, override and validate the YAML configuration.

    Args:
        path: YAML file to load.
        overrides: `dotted.key=value` assignments applied in order.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If an override is malformed.
        pydantic.ValidationError: If the result does not match the schema.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for assignment in overrides or []:
        apply_override(data, assignment)
    return Config.model_validate(data)


def config_hash(config: Config) -> str:
    """Return the sha256 of the canonical JSON form of the resolved config.

    Args:
        config: Validated configuration.

    Returns:
        Hex digest; identical configs always hash identically.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

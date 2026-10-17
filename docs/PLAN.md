# vlm-extract-gepa — Plan

Status: **approved** (2026-10-17), with the decisions recorded in §9.
Scope: P1–P6 as specified in the task brief. This document fixes the structure,
the contracts between modules, and the points that still need a decision (§9).

---

## 0. What was verified before writing this plan

These facts come from reading installed source code or querying live public
endpoints on 2026-10-17. They are not from memory.

| Topic | Finding | Source |
|---|---|---|
| GEPA version | `gepa==0.1.4`. Core has **no dependencies**. `gepa.lm.LM` (string models) uses `litellm`, but we won't use it. | `uv pip install gepa`, `gepa/lm.py` |
| `GEPAAdapter` | Protocol: `evaluate(batch, candidate, capture_traces) -> EvaluationBatch`, `make_reflective_dataset(candidate, eval_batch, components_to_update) -> Mapping[str, Sequence[Mapping]]`, optional `propose_new_texts` attribute (default `None`). Never raise for per-example failures; return score 0. | `gepa/core/adapter.py` |
| `EvaluationBatch` | dataclass `outputs, scores, trajectories=None, objective_scores=None, num_metric_calls=None` | same |
| `reflection_lm` | Either a model string (then wrapped in litellm `LM`) or **any sync callable `(str \| list[dict]) -> str`**. If the callable has a `total_cost` attribute, GEPA uses it as-is, so our own cost tracker plugs straight into `max_reflection_cost`. | `gepa/api.py:468-478`, `proposer/reflective_mutation/base.py` |
| Component selection | `module_selector="round_robin"` (one component per iteration, cycling) or `"all"`, or any callable `(state, trajectories, subsample_scores, candidate_idx, candidate) -> list[str]`. | `strategies/component_selector.py` |
| Merge | `use_merge=False` by default; `max_merge_invocations=5`. Merge combines two Pareto-front candidates component-wise. | `gepa/api.py`, `proposer/merge.py` |
| Budget | `max_metric_calls` counts **one call per (candidate, example) evaluation**. Per iteration: parent eval on minibatch (m) + child eval on minibatch (m) + a full `valset` eval **only if the child is accepted**. Plus one seed eval on the full `valset` at start. Other stoppers: `max_reflection_cost`, `stop_callbacks`, and a `gepa.stop` file in `run_dir`. | `reflective_mutation.py:299-332, 543-574`, `engine.py` |
| Images in reflection | GEPA's **default** proposer already accepts `gepa.Image` objects inside reflective-dataset records and sends them to the reflection LM as OpenAI image parts. So multimodal reflection is possible **without** a custom `propose_new_texts`. | `strategies/instruction_proposal.py` |
| `run_dir` | GEPA writes `gepa_state.bin` (resumable), `run_log.txt`, `candidate_tree.html` and `generated_best_outputs_valset/`. **Reusing an existing `run_dir` resumes the run.** | `core/state.py`, `core/engine.py` |
| Callbacks | `GEPACallback` with `on_iteration_end`, `on_candidate_accepted`, `on_valset_evaluated`, `on_pareto_front_updated`, `on_budget_updated`, … We'll use them for per-iteration logging. | `core/callbacks.py` |
| Qwen on OpenRouter | `qwen/qwen3.8-27b` supports `reasoning`, `response_format`, `structured_outputs`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `seed`. **Reasoning is on by default, effort `xhigh`**, not mandatory, and `"none"` is a supported effort. | `GET /api/v1/models` |
| Qwen providers | **20 providers**, from fp4 to bf16, priced from $0.02 to $0.99/M input. Your quoted $0.24/$2.20 is Parasail (fp8). Tenstorrent and Cerebras **don't support `response_format`**; Novita and Alibaba don't support `structured_outputs`. Most providers cap context at 262k, not 1M. | `GET /api/v1/models/qwen/qwen3.8-27b/endpoints` |
| Haiku on OpenRouter | `anthropic/claude-haiku-5.5`: $0.10/$0.50 per M tokens. **No `temperature`/`top_p`**. Reasoning defaults to effort **`medium`**; `"none"` is supported. Bedrock and AWS endpoints lack `structured_outputs`. | same endpoints |
| Disabling thinking | OpenRouter: `reasoning: {effort: "none"}` ("disables reasoning entirely"). Qwen card: `chat_template_kwargs: {enable_thinking: false}` (vLLM-style, honored only by self-hosted-style providers). P1 will send both and **confirm with a live call** that `usage.completion_tokens_details.reasoning_tokens == 0` and that no `reasoning` field comes back. Needs your `OPENROUTER_API_KEY`; there's none on this machine yet. | openrouter.ai/docs/use-cases/reasoning-tokens, HF card |
| Qwen sampling (non-thinking) | `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0` | https://huggingface.co/Qwen/Qwen3.8-27B (fetched 2026-10-17) |
| Kili SDK | template-kili's docs say 2.176.1, but its `pyproject.toml`, `uv.lock` and Confluence pages pin **`kili==2.142.1`**. Its Confluence says 2.142.1 is your company version. It forces **`pillow<10.1`** (resolves to 10.0.1 alongside everything else we need). `kili.labels()` exposes **`secondsToLabel`**, so review time is recoverable. Assignment uses e-mails; `create_predictions(model_name=...)` takes one model per call; `ParsedLabel` needs `import kili.client` first (circular-import trap). | template-kili `confluence/01-version-et-compatibilite.md`, `.venv/.../kili` |
| Dependency resolution | `kili==2.142.1 + gepa + pymupdf + faker + jinja2 + rapidfuzz + httpx + pydantic-settings + typer + loguru + scikit-learn` resolves on 3.12, as do `+ fastembed` and `+ weasyprint`. | `uv pip compile` dry runs |
| Domain change | Since **1 April 2024**, French insurers no longer issue the green card and windscreen sticker. They issue a **"Mémo Véhicule Assuré"** (content set by art. R211-14-0-1 Code des assurances). The international card ("carte internationale d'assurance automobile", ex-green card) still exists and is now white. Real documents will mix: old green cards, the new white card, Mémos and free-form "attestations". The field descriptions must cover all four. | matmut.fr, groupama.fr, generali.fr (searched) |
| P1 live checks (2026-10-17) | `reasoning: {effort: "none"}` alone gives 0 reasoning tokens on Parasail (`chat_template_kwargs` was redundant, so it's dropped). **`min_p` makes Parasail routing fail** (HTTP 404 under `require_parameters`); the card's value 0.0 is neutral, so it's omitted. Haiku with `effort: medium` + strict `json_schema` works; reasoning is adaptive (0–306 tokens observed, 0 on document extraction). Measured: an extractor call costs **$0.0011–0.0013** (~3.5k input tokens per page), ~$0.0006 when the provider prompt cache hits. A Haiku extraction costs **~$0.0009** (~8.3k input tokens per page). | live calls, `vex doctor` |
| P2 data checks (2026-10-17) | Public specimens: ~3 usable, 1 approved (Leocare Mémo mock-up); the others were excluded for personal data, for being specs rather than certificates, or for blocking bots (Allianz 403). Synthetic set: 200 docs, 5 layouts, 131 scans / 69 clean PDFs, 15 two-page, 10 two-vehicle, 16 rotated ≥ 90°. **Sanity check on 12 scans vs synthetic_meta** (seed prompt): field accuracy 96.7% with `extractor_eval` vs 95.8% with the official params, no parse failures. Q2 is kept for reproducibility; the synthetic set is easy for this model (kept as-is by decision). **Reproducibility:** `kili==2.142.1` was added in P2 (not P4) so Pillow is pinned to 10.0.1 *before* doc_ids are frozen. Pillow 10.0.1's JPEG encoder turned out to be nondeterministic across processes on Apple Silicon (same pixels → different bytes in ~half the runs), so scans are JPEG-encoded by PyMuPDF instead; three full regenerations are byte-identical, and a cross-process regression test guards it. | `vex data *`, scratch scripts |
| Real specimens | **Scarce.** Insurers generate Mémos per customer in the customer area. No official blank specimen PDF turned up in a first search; most public examples are images inside explainer articles, with unclear licences. See open question Q1: this directly threatens "~15 REAL docs in `heldout_blind`". | web search |

---

## 1. Repository tree

```
vlm-extract-gepa/
├── pyproject.toml              # uv project; script entry point `vex = vex.cli:app`
├── uv.lock
├── .python-version             # 3.12
├── .env.example                # OPENROUTER_API_KEY, KILI_API_KEY, KILI_API_ENDPOINT
├── .gitignore                  # .env, data/raw, data/inbox, .cache, runs, *.bin …
├── README.md                   # setup, quick start, annotation-round workflow, Mac vs work
├── configs/
│   └── default.yaml            # every tunable: models, params, paths, splits, weights, GEPA, AL
├── docs/
│   ├── PLAN.md                 # this file
│   ├── gepa_guide.md           # GEPA for beginners + concept→code table (P5)
│   └── kili_uncertainties.md   # what KiliBackend could not verify offline (P4)
├── notebooks/
│   └── gepa_walkthrough.ipynb  # 2-iteration GEPA run on ~5 locally corrected synthetic docs (P5)
├── prompts/                    # committed: versioned candidates  <name>.<hash8>.json
├── data/
│   ├── manifest.jsonl          # committed: one line per real specimen (url, sha256, date, licence)
│   ├── specimen_sources.yaml   # committed: curated list of pages to fetch (reviewed by you)
│   ├── registry.jsonl          # committed: doc_id, origin, path, pages, added_at
│   ├── splits.yaml             # committed: frozen, seeded split assignment per doc_id
│   ├── annotations/            # committed: THE source of truth, *.jsonl (one file per round)
│   ├── review_queue/           # committed: suspect_annotations.jsonl
│   ├── synthetic_meta/         # committed: generator values (tests/sanity checks ONLY)
│   ├── raw/{real,synthetic}/   # git-ignored: document files
│   └── inbox/                  # git-ignored: LocalBackend editable JSON files
├── reports/
│   └── rounds.csv              # committed: per-round tracking (P6)
├── runs/                       # git-ignored: one dir per run (config, hash, logs, GEPA state)
├── .cache/                     # git-ignored: LLM + diagnostic disk caches
├── src/vex/
│   ├── __init__.py
│   ├── cli.py                  # typer app: every `vex <command>`, global `--set` / `--no-cache`
│   ├── config.py               # pydantic YAML models, Settings(.env), `--set` overrides, config hash
│   ├── runs.py                 # create run dir, write resolved config + hash
│   ├── logs.py                 # loguru setup with secret redaction
│   ├── schema.py               # AttestationAuto + field descriptions + strict JSON-schema export
│   ├── prompt.py               # seed candidate from schema, render candidate→prompt, save/load/hash
│   ├── extract.py              # one doc + candidate + role → ExtractionResult (parsed | failed)
│   ├── llm/
│   │   ├── client.py           # async OpenRouter client: passthrough params, structured output, retry, semaphore
│   │   ├── media.py            # PDF→PNG (pymupdf, DPI, max pages), image bytes, data URLs
│   │   ├── cache.py            # sha256-keyed disk cache (JSON files), bypassable
│   │   ├── cost.py             # per-call/per-run usage + USD, BudgetGuard, GEPA stopper
│   │   └── sync.py             # (P5) run async coroutines from sync code (GEPA, Jupyter-safe)
│   ├── data/
│   │   ├── registry.py         # doc_id = sha256(file), origin, registry I/O
│   │   ├── splits.py           # seeded frozen split assignment; split views filtered by gold
│   │   ├── specimens.py        # fetch real specimens: robots.txt, manifest, PII skip list
│   │   └── synth/
│   │       ├── generator.py    # seeded Faker fr_FR values → Jinja2 HTML → PDF/PNG
│   │       ├── logos.py        # procedurally drawn FICTIONAL insurer logos (Pillow)
│   │       ├── augment.py      # rotation, blur, JPEG artefacts, contrast, noise
│   │       └── templates/      # ≥5 *.html.j2 layouts (mémo, white card, legacy green card, attestation, letter)
│   ├── metric/
│   │   ├── normalize.py        # dates, amounts, text, plates, lists
│   │   ├── compare.py          # per-field exact / fuzzy / set-F1 + error type + diff text
│   │   └── report.py           # doc score, per-field & global scores, error breakdown
│   ├── annotation/
│   │   ├── store.py            # append-only JSONL store; gold = latest reviewed|blind per doc
│   │   ├── backend.py          # AnnotationBackend protocol + factory from config
│   │   ├── local.py            # LocalBackend: inbox JSON files
│   │   ├── kili_payloads.py    # PURE: json_interface, json_response, metadata, label→record
│   │   └── kili_backend.py     # KiliBackend: SDK calls (UNTESTED against a live instance)
│   ├── bootstrap.py            # zero-shot round → corrections → proposed description diff → confirm
│   ├── diagnostic.py           # diagnostic VLM call, cache, suspect-annotation queue
│   ├── gepa_adapter.py         # GEPAAdapter implementation (heavily commented)
│   ├── optimize.py             # gepa.optimize wiring, reflection_lm wrapper, callbacks, held-out scoring
│   ├── active/
│   │   ├── signals.py          # (a) Pareto disagreement (b) self-consistency (c) rules (d) verifier
│   │   ├── diversity.py        # embeddings (optional group) + KMeans + cross-cluster pick
│   │   └── select.py           # combine signals, add random confident sample, push round N+1
│   └── tracking.py             # rounds.csv rows, `vex report`, plateau detection
└── tests/
    ├── conftest.py             # fake OpenRouter transport (httpx.MockTransport), tmp dirs
    ├── fixtures/               # tiny PDFs/PNGs, canned model responses
    ├── test_config.py  test_client.py  test_cache.py  test_cost.py  test_schema.py  test_prompt.py
    ├── test_registry.py  test_splits.py  test_specimens.py  test_synth.py
    ├── test_normalize.py  test_compare.py  test_report.py
    ├── test_store.py  test_local_backend.py  test_kili_payloads.py  test_bootstrap.py
    ├── test_gepa_adapter.py  test_optimize.py  test_diagnostic.py
    ├── test_signals.py  test_select.py  test_tracking.py
    └── test_live.py            # @pytest.mark.live — ONE extractor call through OpenRouter
```

Package name `vex` (short imports, same as the CLI). `scripts/` is not needed:
the specimen fetcher is `vex data fetch-specimens`, so it stays testable.

---

## 2. Key design decisions

1. **HTTP client: `httpx` directly, not the `openai` SDK.** The brief says `params`
   and `extra_body` are passed through *unchanged*. The `openai` SDK rejects
   unknown kwargs like `top_k`, `min_p` and `repetition_penalty` unless they're moved
   into its own `extra_body`. With httpx the request body is literally
   `{model, messages, response_format?, **params, **extra_body}`. Fewer layers, no surprises.
2. **Reasoning: off for the extractor, on for Haiku (§9 Q4).** The extractor gets
   `extra_body: {reasoning: {effort: "none"}, chat_template_kwargs: {enable_thinking: false}}`.
   P1's live check confirms that `reasoning_tokens == 0`. The three Haiku roles get
   `reasoning: {effort: "medium"}` (OpenRouter's default for this model, written out
   explicitly), with `max_tokens` raised so thinking can't truncate the JSON answer.
   Reasoning tokens are logged separately in the cost tracker.
3. **Provider pinning through `extra_body.provider`** (OpenRouter routing, passed
   through unchanged): `{order: [...], allow_fallbacks: false, require_parameters: true}`.
   `require_parameters: true` stops OpenRouter from silently routing to a provider
   that ignores `response_format`. Pinning also keeps cached results and GEPA scores
   comparable (fp4 vs bf16 outputs differ). Which provider to pin is Q3.
4. **Structured output.** `structured_output: auto | json_schema | prompt` per role.
   `auto` reads the pinned endpoint's `supported_parameters` once per run (cached).
   If `structured_outputs` is listed, it sends `response_format: {type: json_schema, json_schema: {name, strict: true, schema}}`;
   otherwise it falls back to prompt-only JSON plus parsing. Either way, a parse or
   validation failure triggers **one repair retry** (raw output + validation error
   sent back). A second failure gives `ExtractionResult(failed=True)`: document score 0
   and every field typed `omission`. The path used is logged and stored in the trajectory.
   The schema sent is a strict variant of `AttestationAuto.model_json_schema()`
   (all properties required, `anyOf [T, null]`, `additionalProperties: false`),
   which strict mode requires.
5. **Cache key** = sha256 of canonical JSON `(model, params, extra_body, response_format, messages)`,
   with each image part replaced by `sha256(image bytes)`. Self-consistency passes
   (P6) add a `sample_index` to the key; otherwise a temperature > 0 pass would just
   replay the cached sample. Bypass with `--no-cache` (run) or `use_cache=False` (call).
   The cache stores the raw response, so cost is re-reported as `cached=True, usd=0`.
6. **Cost.** Use `usage.cost` from the OpenRouter response when present (verified
   in P1). Otherwise compute from a YAML price table. `BudgetGuard` raises
   `BudgetExceeded` *before* sending a call once `budget.max_usd` is reached. In GEPA,
   a `BudgetStopper` (GEPA `StopperProtocol`) checks the same tracker between
   iterations, so the run stops cleanly with its state saved instead of crashing mid-iteration.
7. **Async inside sync GEPA.** GEPA calls `evaluate()` synchronously. `llm/sync.py`
   owns one long-lived event loop on a background thread, and `evaluate` submits
   `gather(...)` to it. This also works in Jupyter, where a loop is already running
   and `asyncio.run` would fail.
8. **Component selector: error-driven, not round-robin (recommendation, Q6).** With
   11 components, round-robin often selects a field that is *correct* on the current
   minibatch. Its reflective dataset is then empty, and GEPA still spends 2·m metric
   calls testing a blind rewrite. A ~20-line selector picks the failing `field:*`
   component with the most weighted error in `trajectories`, or `instructions` when
   the errors are parse failures or spread across fields. `round_robin` and `all`
   remain one YAML switch away.
9. **Diagnostic cache key** = `(doc_id, field, sha(pred), sha(gold))` as specified,
   **plus the diagnostic model id and prompt version**. Otherwise swapping the
   diagnostic model in YAML would silently serve stale verdicts.
10. **Synthetic renderer: PyMuPDF `fitz.Story`** (HTML + CSS subset → PDF), then
    rasterized with PyMuPDF and augmented with Pillow + numpy.
    Why this over WeasyPrint or Playwright: it's already a dependency, needs no system
    libraries (WeasyPrint needs `brew install pango`, which isn't present here) and
    no 150 MB browser download (Playwright). It's also deterministic, and it will work
    on a locked-down work laptop. The cost: limited CSS (no flex/grid), so layouts use
    tables and fixed blocks, which matches how these forms actually look. If that proves
    too limiting for a layout, switching to WeasyPrint only touches `generator.py` (Q7).
11. **Kili project input type = `PDF`.** PNG/JPEG docs are wrapped into a single-page
    PDF with PyMuPDF before upload, so one project holds every document.
    `external_id = doc_id`.

---

## 3. Phases and definition of done

Every phase ends with: `uv run pytest` green (live test excluded by default),
`uv run vex <command>` working on this Mac, a short "verified / not verified"
note in the commit message body, and **one local commit**. No push.

### P1 — Config, VLM client, schema, prompt-as-candidate
- `configs/default.yaml` + `config.py`: pydantic models (§4), `Settings` from `.env`,
  `--set a.b.c=value` (value parsed as YAML scalar/list/dict, unknown keys rejected),
  `config_hash = sha256(canonical JSON of resolved config)`, secrets excluded.
- `runs.py`: `runs/<UTC timestamp>-<command>-<hash8>/{config.resolved.yaml, config.sha256, log.txt, cost.json}`.
- `llm/*`: client, media, cache, cost, sync (decisions 1–7).
- `schema.py`: `AttestationAuto` with detailed `Field(description=...)`, including the
  multi-vehicle rule (first vehicle only, stated in `immatriculation` and
  `marque_vehicule`) and the four document families. Validators turn `""` into `None`.
- `prompt.py`: `seed_candidate()` → `{"instructions": ..., "field:assureur": ..., ...}`;
  `render(candidate) -> (system_prompt, json_schema)`, where field descriptions in the
  rendered JSON schema come **from the candidate**, not the class; `save(candidate, name)` → `prompts/<name>.<hash8>.json`.
- **CLI:** `vex config show [--set ...]`, `vex prompt seed`, `vex extract <file> [--candidate prompts/x.json] [--role extractor] [--no-cache]`,
  `vex doctor` (key present, pinned providers still serve the model with the required parameters).
- **Live check (requires your key):** one extractor call. Confirm extractor reasoning
  tokens = 0, the structured-output path and the cost field. Measure Haiku's
  reasoning tokens at `medium` to replace the §8 assumption, and compare official vs
  `extractor_eval` params on 2–3 docs. Record the results in the P1 commit.
- **Tests:** override parsing, hashing stability, passthrough body equality,
  backoff on 429/503 (mocked), semaphore limit, cache hit/miss/bypass, budget stop,
  JSON repair path, PDF render page cap, strict-schema shape, candidate round trip.

### P2 — Data
- `vex data fetch-specimens`: reads `data/specimen_sources.yaml` (curated URLs, each with
  a licence/ToS note), checks `robots.txt` (`urllib.robotparser`), uses a polite
  delay and a custom UA, then downloads into `data/raw/real/` and appends to `manifest.jsonl`.
  Only entries you set to `status: approved` are fetched (review happens in the YAML, so
  no separate approve command); `pii: suspected` is refused even if approved.
- `vex data synth --n 200 --seed 42`: ≥5 layouts, fictional insurers and logos,
  scan augmentation, values → `data/synthetic_meta/<doc_id>.json`. The store
  refuses any record whose annotator is `synthetic_meta`, and a test asserts that
  `synthetic_meta` is never read outside tests and `vex data sanity`.
- `vex data register` (scan `data/raw`, fill `registry.jsonl`), `vex data split` (§6).
- **Tests:** generator determinism (same seed → same bytes), ≥5 layouts, augmentation
  bounds, robots.txt refusal (mocked), manifest schema, split invariants
  (heldout real-only, frozen, no overlap).

### P3 — Metric
- `normalize.py`: dates (`01/02/2025`, `1/2/25`, `1er février 2025`, `01.02.2025`,
  `2025-02-01`, `du 01/02/2025 au …` → handled by field), amounts (`1 234,56 €` → `1234.56`),
  text (casefold, strip accents, collapse whitespace, strip punctuation), plates
  (SIV `AA-123-AA`, FNI `1234 AB 56` / `123 ABC 75` → canonical uppercase with fixed separators),
  lists (normalized `frozenset`).
- `compare.py`: exact by default; rapidfuzz `token_sort_ratio ≥ threshold` for the YAML
  `fuzzy_fields`; set-F1 for `garanties`; error types; diff string
  `"[wrong_value] immatriculation: expected 'AB-123-CD' got 'AB-123-CO'"`.
- `report.py`: weighted doc score, per-field accuracy, error-type breakdown, JSON + table.
- **CLI (changed in P3):** `vex data sanity --n 20 [--candidate X]` scores a candidate against
  synthetic_meta on GEPA-bucket synthetic docs only (sanity, never labels; `vex.data.sanity` is
  synthetic_meta's only reader, guarded by a test). `vex eval` needs gold from the annotation
  store, so it moves to P4.
- **Fuzzy matching:** max(`token_sort_ratio`, `ratio` with spaces removed) on normalized text,
  so a whitespace-only difference ("Pottiernec" / "Pottier nec") is not an error.
- **Live sanity (seed prompt, 20 docs, $0.028):** 98.6% weighted. All 4 errors are `garanties`
  hallucinations on Mémos: the "ASSISTANCE 24H/24" phone block read as a guarantee, a real
  ambiguity for GEPA to fix.
- **Tests:** a table of ~60 normalization cases, every error type, weights, set-F1 edge
  cases (both null = 1, gold null + pred list = hallucination), parse-failure → all omission.

### P4 — Annotation store and backends
- `vex eval --candidate X --split gepa_train|gepa_pareto` (moved from P3; refuses held-out splits).
- **As built:** `vex bootstrap --round N` is one stateful command. The first call selects,
  pre-annotates and pushes; later calls pull and report progress, and once the round is
  complete they propose the rewrite (diff + confirmation, `--yes` to skip). The candidate for
  round N defaults to `bootstrap-r<N-1>`. Also `vex annotate push-blind` (held-out + stand-in,
  round 0, no predictions) and `vex annotate pull --round N`. Local inbox files are named
  by the full doc_id. The store is append-only with an idempotent `append_new`, because Kili
  re-returns every label on each pull. Kili checklist: `docs/kili_uncertainties.md`.
- **Live check (scratch dirs, real store left empty):** round-1 push pre-annotated 5 docs for
  $0.0047, the real Leocare Mémo first and correctly read.
- `store.py`: append-only JSONL (§5). `gold(doc_id)` = latest record with
  `status ∈ {reviewed, blind}`. `pre_annotated` is never returned as gold (tested).
- `LocalBackend`: `push` writes `data/inbox/round_<N>/<doc_id>.json` (prefilled, or empty
  for blind) plus a link to the document file. `pull` imports edited files.
  It considers a file done when you set `"_done": true`, so half-edited files are never imported.
- `kili_payloads.py` (pure, 2.142.1 shapes): `json_interface` with one `TRANSCRIPTION` job
  per field (`required: 0`, `garanties` = one per line) + `COMMENTAIRE` job;
  `prediction_json_response(values)` (null → job omitted, as template-kili example 07 does);
  `asset_metadata(doc_id, status, round)`; `label_to_record(label)` (empty text → null,
  `secondsToLabel` → `review_seconds`, `author.email` → annotator).
- `kili_backend.py`: `create_project`, `append_many_to_dataset`, `create_predictions`
  (skipped for blind docs), `labels(type_in=[DEFAULT, REVIEW])`. It never runs here:
  instantiation fails fast unless `annotation.backend == "kili"` and the credentials exist.
  Everything is marked **UNTESTED LIVE** in docstrings and in `docs/kili_uncertainties.md`.
- `bootstrap.py` + **CLI** `vex bootstrap --round 1|2|3` (5 → 10 → 15 docs):
  pre-annotate with `preannotator` → push → (you edit) → `vex annotate pull` → the
  `reflection` model proposes new `field:*` texts from corrections + comments →
  a coloured unified diff → `Apply? [y/N]` → save the new candidate to `prompts/`.
  `vex annotate push-blind` pushes `heldout_blind` with no predictions.
- **Tests:** store invariants, local push/pull round trip, payloads validated with
  Kili's own `ParsedJobs` parser offline (same technique as template-kili), and a guard
  that the bootstrap rewriter's input never contains a heldout doc_id.

### P5 — GEPA adapter and optimize
- `gepa_adapter.py`: `VexAdapter.evaluate` → `EvaluationBatch(outputs, scores, trajectories)`,
  with trajectory `{doc_id, raw_output, parsed, structured_path, field_results[]}`.
  `make_reflective_dataset` → diagnostics (failing fields only, cached, quote mandatory,
  `located=false` → `suspect_annotations.jsonl`), records routed to `field:<name>`
  and an aggregated record to `instructions`. Record format follows GEPA's recommended
  `{"Inputs", "Generated Outputs", "Feedback"}`. A documented `propose_new_texts = None`
  extension point explains how a multimodal proposer would plug in.
- `optimize.py` + **CLI** `vex optimize [--max-metric-calls N] [--resume RUN_DIR]`:
  - trainset = `gepa_train ∩ gold`, valset = `gepa_pareto ∩ gold`;
  - `reflection_lm` = sync wrapper on our client exposing `total_cost`;
  - `seed`, `use_merge`, `module_selector`, `reflection_minibatch_size`,
    `max_metric_calls`, `max_reflection_cost` all come from YAML;
  - callbacks write `iterations.jsonl` (score, accepted, component, metric calls, USD);
  - outputs: `candidates/*.json`, `pareto_front.json`, `iterations.jsonl`, `cost.json`,
    GEPA's own files, and the best candidate → `prompts/`;
  - **the end of the run is the only place that touches `heldout_blind`:** it scores
    the seed, the best candidate and the top-K Pareto candidates there, then writes
    `heldout.json` + a `rounds.csv` row.
- `docs/gepa_guide.md`, `notebooks/gepa_walkthrough.ipynb` (uses `LocalBackend` gold on
  ~5 synthetic docs that you, or I with your OK, correct by hand; the cost is cents).
- **Tests:** adapter contract (lengths, never raises, parse failure = 0), routing,
  diagnostic dropping and suspect logging, a guard that heldout ids reach neither
  `gepa.optimize` nor the diagnostic or reflection models (spy on the client), and an
  end-to-end `gepa.optimize` with a fake client (2 iterations, deterministic).

### P6 — Active learning and tracking
- `signals.py`: (a) field-level disagreement across the top-K Pareto candidates;
  (b) N passes at `temperature > 0` (sample_index in the cache key) + extractor-vs-Haiku
  disagreement; (c) rules: `date_fin > date_debut`, plate regex, required fields,
  dates within [today − 5 y, today + 2 y]; (d) verifier: the diagnostic model checks
  "is this value in the document? quote it". (d) runs only on the top
  `active.verifier_shortlist` docs by (a)+(b)+(c), to cap cost.
- `diversity.py`: optional group `embed` (`fastembed` CLIP-vision ONNX, no torch).
  Without it, a fallback uses 32×32 grayscale thumbnails. Then scikit-learn KMeans
  and round-robin picks across clusters by score.
- `select.py` + **CLI** `vex select --round N+1 [--size 20]`: combined weighted score +
  `active.random_confident_fraction` random picks from the low-uncertainty tail → push as
  `pre_annotated`, round N+1.
- `tracking.py` + **CLI** `vex report`: `reports/rounds.csv` (§5.2) + plateau flag
  when the held-out gain is below `tracking.plateau_delta` for `tracking.plateau_rounds` rounds.
- **Tests:** each signal on hand-built inputs, weight combination, cluster coverage,
  random sample size, CSV schema, plateau detection.

---

## 4. Config schema (`configs/default.yaml`)

```yaml
models:
  extractor:
    model: qwen/qwen3.8-27b
    structured_output: auto            # auto | json_schema | prompt
    # Official non-thinking sampling params, from https://huggingface.co/Qwen/Qwen3.8-27B
    # (fetched 2026-10-17). Used for self-consistency sampling (P6).
    params: {temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0,
             presence_penalty: 1.5, repetition_penalty: 1.0, max_tokens: 1500, seed: 0}
    extra_body:
      reasoning: {effort: none}
      chat_template_kwargs: {enable_thinking: false}
      provider: {order: [parasail], allow_fallbacks: false, require_parameters: true}  # Q3
  # Q2: deterministic override, merged over `extractor`, used for GEPA scoring,
  # `vex eval` and held-out scoring.
  extractor_eval:
    params: {temperature: 0.0, presence_penalty: 0.0}
  preannotator: &haiku
    model: anthropic/claude-haiku-5.5
    structured_output: auto
    params: {max_tokens: 8000}         # Haiku 5.5 rejects temperature/top_p; room for thinking
    extra_body: {reasoning: {effort: medium}}                                           # Q4
  diagnostic: *haiku
  reflection: {<<: *haiku, params: {max_tokens: 12000}}
pricing_fallback_usd_per_m:            # used only if the response lacks usage.cost
  qwen/qwen3.8-27b: {input: 0.24, output: 2.20}
  anthropic/claude-haiku-5.5: {input: 0.10, output: 0.50}
render: {dpi: 150, max_pages: 3, image_format: png}
client: {concurrency: 8, max_retries: 5, backoff_base_s: 1.0, backoff_max_s: 60, timeout_s: 120}
cache: {enabled: true, dir: .cache/llm}
budget: {max_usd: 5.0}
paths: {raw: data/raw, registry: data/registry.jsonl, splits: data/splits.yaml,
        annotations: data/annotations, inbox: data/inbox, prompts: prompts,
        runs: runs, review_queue: data/review_queue, reports: reports}
splits:
  seed: 42
  heldout_blind_size: 15               # real docs only
  pareto_fraction: 0.4                 # of the non-heldout pool → gepa_pareto bucket
metric:
  weights: {assureur: 1, numero_contrat: 2, nom_souscripteur: 1, adresse_souscripteur: 0.5,
            immatriculation: 3, marque_vehicule: 0.5, date_debut_validite: 2,
            date_fin_validite: 2, code_assureur: 1, garanties: 1}
  fuzzy_fields: [nom_souscripteur, adresse_souscripteur, assureur]
  fuzzy_threshold: 90
annotation:
  backend: local                       # local | kili
  kili: {project_id: null, project_title: "vex - attestations auto", model_name_prefix: vex}
bootstrap: {batch_sizes: [5, 10, 15]}
gepa:
  seed: 0
  max_metric_calls: 300
  max_reflection_cost_usd: 1.0
  reflection_minibatch_size: 3
  candidate_selection_strategy: pareto
  module_selector: error_driven        # error_driven | round_robin | all   (Q6)
  use_merge: true
  max_merge_invocations: 5
  skip_perfect_score: true
  cache_evaluation: true
  heldout_top_k: 3
  reflection_images: false             # Q8
active:
  weights: {pareto_disagreement: 0.35, self_consistency: 0.25, rules: 0.2, verifier: 0.2}
  top_k_candidates: 3
  self_consistency: {n: 3, temperature: 0.9}
  verifier_shortlist: 40
  diversity: {enabled: true, n_clusters: 8, embedder: auto}   # auto = fastembed if installed
  random_confident_fraction: 0.15
  batch_size: 20
tracking: {plateau_delta: 0.01, plateau_rounds: 2}
```

`.env` (pydantic-settings): `OPENROUTER_API_KEY`, `KILI_API_KEY`, `KILI_API_ENDPOINT`.
They're `SecretStr`, excluded from the resolved config and the hash, and never logged.
A loguru filter also redacts `sk-or-…` patterns.

---

## 5. Data schemas

### 5.1 Annotation record (`data/annotations/round_<N>.jsonl`)

```json
{
  "record_id": "uuid4",
  "doc_id": "sha256 of the file",
  "round": 2,
  "status": "pre_annotated | reviewed | blind",
  "values": {"assureur": "…", "garanties": ["RC", "Vol"], "code_assureur": null, "…": "…"},
  "source": "model | human",
  "annotator": "model:anthropic/claude-haiku-5.5 | local:antoine | kili:<email>",
  "review_seconds": 94.0,
  "comment": "two vehicles on the certificate, kept the first one",
  "candidate_hash": "sha256-8 of the candidate used for pre-annotation (null for blind)",
  "model_id": "anthropic/claude-haiku-5.5 (null for human records)",
  "backend": "local | kili",
  "created_at": "2026-10-17T14:03:00Z",
  "reviewed_at": "2026-10-18T09:12:44Z"
}
```

Only `reviewed` and `blind` count as gold. Records are append-only; a later record
for the same doc supersedes an earlier one (corrections of suspect annotations
are just a new `reviewed` record).

### 5.2 `reports/rounds.csv`

`round, date, candidate_hash, heldout_global, heldout_<field>…(10), n_gold_train,
n_gold_pareto, fields_corrected_share, median_review_seconds, suspect_annotations,
usd_round, usd_cumulative, plateau`

---

## 6. Split strategy

- `doc_id = sha256(file bytes)`; `origin ∈ {real, synthetic}` in `registry.jsonl`.
- `vex data split` runs **once**, seeded, and writes `data/splits.yaml`:
  1. `heldout_blind`: `heldout_blind_size` docs drawn from **real** docs only.
  2. Every other doc gets a **bucket**, `gepa_train` or `gepa_pareto`
     (`pareto_fraction`), stratified by origin.
  Later additions (new specimens) are appended with the same rule; existing
  assignments never move. The file is committed.
- **Effective** splits are computed, not stored: `gepa_train = bucket_train ∩ gold`,
  `gepa_pareto = bucket_pareto ∩ gold`. So the sets grow round by round while
  assignments stay frozen, and an unreviewed doc can never enter them.
- **Mac stand-in (§9 Q1):** `heldout_standin` holds 15 synthetic documents of a reserved
  layout (`carte_verte`); the rest of that layout sits in `reserved_unused`. Both are treated
  exactly like `heldout_blind` by every guard. Layouts come from the file name, never from
  synthetic_meta.
- `heldout_blind` is annotated blind (no predictions) and **only ever scored, never
  learned from**. Two read-only entry points may load it: the end of `vex optimize`
  and `vex report` (per-round score of the current candidate, §9 Q5). The adapter,
  bootstrap rewriter, diagnostic and active-learning entry points assert their inputs
  are disjoint from it, and a test spies on the client to check that no held-out
  doc or score reaches a model call that proposes text.
- Caveat, documented in the guide: use the per-round held-out scores to decide
  *when to stop*, never to *choose between* candidates. Otherwise you slowly
  select on the held-out set too.
- `docs/gepa_guide.md` explains the mapping: GEPA's `valset` is **not** held out (it
  picks candidates, so scores on it are optimistic). Your "validation" is `heldout_blind`.

---

## 7. The annotation-round workflow (end to end)

```
round 0  vex data fetch-specimens / synth / register / split
         vex annotate push-blind               → you annotate heldout_blind blind
round 1  vex bootstrap --round 1  (5 docs)     → pre-annotate → push → correct → pull
                                               → reflection proposes field texts → diff → y/N
round 2  vex bootstrap --round 2  (10 docs)    → same
round 3  vex bootstrap --round 3  (15 docs)    → same   (~30 gold docs now)
round 4+ vex optimize                          → best candidate + heldout score + rounds.csv row
         vex select --round 5                  → uncertainty+diversity batch pushed pre-annotated
         (correct, pull) → vex optimize → vex report → stop at plateau
```

---

## 8. Cost estimates (USD)

Assumptions, to be replaced by measured numbers after P1's live call:
- One page at 150 DPI is ~1240×1754 px. Qwen VL is ~32 px per token, so ~2.2k image
  tokens. Claude resizes to a 1568 px long edge, so ~2.3k tokens. Prompt + schema is
  ~1.5k tokens, so input is **~4k tokens/page**. Output is ~300 tokens of JSON.
- Extractor call ≈ 4k×$0.24/M + 300×$2.20/M ≈ **$0.0016**.
- Haiku runs with reasoning at `medium`. I assume **~1.5k thinking tokens per call**
  (billed as output, +$0.00075), to be replaced by the P1 measurement.
  Haiku extraction ≈ 4k×$0.10/M + 1.8k×$0.50/M ≈ **$0.0013**; diagnostic ≈ **$0.0012**;
  reflection (8k text in, 2.5k out) ≈ **$0.0021**.
- Extractor reasoning is **off**. Left at its default `xhigh`, it would add several
  thousand output tokens per call: roughly **5–20× the extractor cost**.

| Operation | Calls | USD |
|---|---|---|
| Bootstrap round (15 docs): pre-annotation | 15 Haiku | ~$0.02 |
| … description rewrite | 1–3 reflection | <$0.01 |
| **One bootstrap round** | | **≈ $0.03** (your review time dominates: ~15 × 2–4 min) |
| GEPA run, `max_metric_calls=300`, \|pareto\|=20, m=3, ~30% acceptance → ~23 iterations | 300 extractor | $0.48 |
| … diagnostics (≈9 failing fields/iteration, many cache hits) | ≤200 | ≤$0.24 |
| … reflection | ~23 | $0.05 |
| … final held-out scoring (seed + best + top-3, 15 docs) | 75 extractor | $0.12 |
| **One GEPA run (300 calls)** | | **≈ $0.90** (≈ $2.70 at 1000 calls; ×1.5–2 if docs average 2 pages) |
| `vex select` on a 200-doc pool (K=3, N=3, shortlist 40) | ~1200 extractor + 200 Haiku + ~400 diag | ≈ $2.70 |
| `vex report` held-out scoring of the current candidate (Q5) | 15 extractor | $0.03 |

`budget.max_usd: 5.0` covers a full run plus selection with margin. The disk cache
makes repeated evaluations of the same (candidate, doc) free.

---

## 9. Decisions (approved 2026-10-17)

1. **Real specimens.** Images embedded in articles are accepted only from insurer or
   official sources, with the page URL and a "© source, internal research use" note in the
   manifest; anything showing a plausibly real name or plate is skipped. `heldout_blind`
   stays real-only: half of the real docs, capped at `heldout_blind_size`, with a
   low-confidence warning below 8. **Update (P2, 2026-10-17):** public research found
   only ~3 usable specimens, and only the Leocare Mémo mock-up was approved. So on the Mac
   `heldout_blind` is empty, and a **synthetic stand-in** is used instead: the `carte_verte`
   layout is reserved (`splits.standin_layouts`), 15 of its documents form
   `heldout_standin` (annotated blind, reported apart from `heldout_blind`), and the other
   25 are parked in `reserved_unused`. A reserved layout never reaches GEPA. It measures
   generalization to an unseen layout, not real-world accuracy. At work: set
   `standin_layouts: []` and add anonymized company documents, which fill `heldout_blind`.
2. **Sampling.** The official Qwen non-thinking params stay in YAML (used for P6
   self-consistency). A deterministic `extractor_eval` override (`temperature: 0`,
   `presence_penalty: 0`, fixed `seed`) is used for GEPA scoring, `vex eval` and
   held-out scoring. P1's live check compares both on 2–3 docs.
3. **Provider.** Parasail pinned, `allow_fallbacks: false`, `require_parameters: true`.
   A work config overlay adds `provider.data_collection: deny`, and `vex doctor`
   prints which pinned providers survive that filter.
4. **Haiku reasoning: enabled** on all three Haiku roles, at `effort: medium`
   (explicit), with `max_tokens` 8000 (12000 for reflection). Reasoning tokens are
   tracked separately. Extractor reasoning stays off.
5. **Per-round held-out score.** `vex report` scores the current candidate on
   `heldout_blind` each round (read-only, results only to `rounds.csv`). Guard and
   caveat: see §6.
6. **Component selector:** error-driven by default (§2.8); `round_robin` / `all` via YAML.
7. **Renderer:** PyMuPDF `Story` (§2.10).
8. **Images in reflection:** off (`reflection_images: false`), documented in the guide
   as the cheapest next step.
9. **"Amounts" normalization:** implemented and tested as required; no schema field uses it yet.
10. **Parse failure scoring:** after the repair retry, every field is typed `omission`
    and the document scores 0.

---

## 10. Risks

- **Kili 2.142.1 is unverified live** (template-kili was never run against a server).
  Shapes are validated offline with `ParsedJobs`. `docs/kili_uncertainties.md`
  lists the checks to run first at work (external_id length with a 64-char sha,
  blank TRANSCRIPTION export shape, `secondsToLabel` on reviewed labels, local-path upload on-prem).
- **Pillow is pinned to 10.0.1** because of Kili. That's fine for augmentation, but it
  constrains future dependencies. If it becomes a blocker, Kili moves to an optional group with a uv conflict declaration.
- **Small gold sets** (~30 docs before GEPA) make Pareto selection noisy. The guide
  explains how to read that and why `heldout_blind` is the arbiter.

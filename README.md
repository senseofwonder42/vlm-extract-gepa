# vlm-extract-gepa

Extract structured fields from French motor insurance certificates (attestations,
Mémo Véhicule Assuré, international insurance cards) with a VLM, and optimize the
extraction prompt with [GEPA](https://github.com/gepa-ai/gepa). The ground truth
is built at the same time, by human review.

> Work in progress. The design is in [docs/PLAN.md](docs/PLAN.md). This README
> covers what exists so far (P1); the full annotation workflow is added with P4–P6.

## Setup

```bash
uv sync
cp .env.example .env   # then set OPENROUTER_API_KEY
uv run vex doctor      # checks the key and that pinned providers accept our parameters
```

## Quick start

```bash
uv run vex config show                                   # resolved config + hash
uv run vex prompt seed                                   # seed candidate → prompts/seed.<hash>.json
uv run vex extract path/to/attestation.pdf               # deterministic extractor (extractor_eval)
uv run vex --set models.extractor.params.temperature=0.2 extract doc.pdf --role extractor
uv run vex --no-cache extract doc.pdf                    # bypass the LLM disk cache
```

Every model call is configured in [configs/default.yaml](configs/default.yaml):
`params` and `extra_body` are sent to OpenRouter unchanged. Any YAML key can be
overridden with `--set dotted.key=value`. Each `extract` run writes
`runs/<timestamp>-extract-<hash>/` containing the resolved config, its hash,
the log, `result.json` and `cost.json`.

## Tests

```bash
uv run pytest            # offline; all network calls are mocked
uv run pytest -m live    # one real extractor call through OpenRouter (< $0.01)
```

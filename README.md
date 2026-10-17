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

## Data

```bash
uv run vex data fetch-specimens   # approved entries of data/specimen_sources.yaml → data/raw/real/
uv run vex data synth             # 200 seeded synthetic certificates → data/raw/synthetic/
uv run vex data register          # add new files to data/registry.jsonl (doc_id = sha256)
uv run vex data split             # frozen assignment in data/splits.yaml
```

```bash
uv run vex data sanity --n 20     # score the seed prompt against generator values (sanity only)
```

Raw documents are git-ignored. The manifest, registry, splits and the
generator's `synthetic_meta` (used only by tests and sanity checks, never as labels) are
committed, so `fetch-specimens` + `synth` rebuild the same files byte-for-byte.
On the Mac, the `carte_verte` layout is reserved as a held-out stand-in
(`splits.standin_layouts`); at work, set it to `[]` and add real documents.

## Annotation rounds

```bash
uv run vex annotate push-blind        # held-out (+ Mac stand-in) docs, no predictions, round 0
uv run vex annotate pull --round 0    # import the blind annotations once done

uv run vex bootstrap --round 1        # 5 docs: pre-annotate (Haiku) + push
#   ... correct them (local: edit data/inbox/round_1/*.json, set "_done": true) ...
uv run vex bootstrap --round 1        # pull; when complete, propose a rewrite of the
                                      # field descriptions → diff → confirm → prompts/bootstrap-r1.*.json
uv run vex bootstrap --round 2        # 10 docs, starts from bootstrap-r1
uv run vex bootstrap --round 3        # 15 docs

uv run vex eval --split gepa_pareto --candidate prompts/bootstrap-r3.<hash>.json
```

`annotation.backend: local` (default, this Mac) uses editable JSON files in
`data/inbox/`. At work, set `annotation.backend: kili` and fill `KILI_API_KEY` /
`KILI_API_ENDPOINT` in `.env`. The Kili backend has **not** been run against a
live instance: read [docs/kili_uncertainties.md](docs/kili_uncertainties.md) first.
Every annotation lands in `data/annotations/round_<N>.jsonl`, the single source
of truth. Only `reviewed` and `blind` records count as gold.

## Optimizing the prompt with GEPA

New to GEPA? Read [docs/gepa_guide.md](docs/gepa_guide.md), then run the 2-iteration
walkthrough [notebooks/gepa_walkthrough.ipynb](notebooks/gepa_walkthrough.ipynb)
(`uv run --with jupyter jupyter lab`, or open it in VS Code; it costs a few cents and
uses a separate, clearly labelled demo store, never `data/annotations/`).

```bash
uv run vex optimize                          # seed = latest prompts/bootstrap-r*, else the schema seed
uv run vex optimize --max-metric-calls 100   # smaller budget
uv run vex optimize --resume runs/<dir>      # continue an interrupted run
```

GEPA trains on `gepa_train` ∩ gold and selects on `gepa_pareto` ∩ gold. After it finishes,
the seed, the best and the top Pareto candidates are scored on `heldout_blind` (and the Mac
stand-in `heldout_standin`), the only place those splits are used besides `vex report`.
Outputs: `runs/<timestamp>-optimize-<hash>/` and `prompts/gepa-best.<hash>.json`.

## Tests

```bash
uv run pytest            # offline; all network calls are mocked
uv run pytest -m live    # one real extractor call through OpenRouter (< $0.01)
```

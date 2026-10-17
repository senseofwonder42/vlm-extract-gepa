# vlm-extract-gepa

Extract structured fields from French motor insurance certificates (attestations,
Mémo Véhicule Assuré, international insurance cards, old green cards) with a VLM,
and optimize the extraction prompt with [GEPA](https://github.com/gepa-ai/gepa).
There are no labels at the start: the ground truth (human-reviewed) and the
optimized prompt are built side by side, round after round, guided by evaluation.

- Design and decisions: [docs/PLAN.md](docs/PLAN.md)
- GEPA explained for beginners: [docs/gepa_guide.md](docs/gepa_guide.md) and
  [notebooks/gepa_walkthrough.ipynb](notebooks/gepa_walkthrough.ipynb)
- Kili (work) checklist: [docs/kili_uncertainties.md](docs/kili_uncertainties.md)
- Documentation détaillée en français : [docs/fr/](docs/fr/README.md)

## Setup

```bash
uv sync                    # Python 3.12, all dependencies
uv sync --extra embed      # optional: ONNX image embeddings for `vex select` diversity
cp .env.example .env       # set OPENROUTER_API_KEY (and KILI_* at work)
uv run vex doctor          # checks the key and that pinned providers accept our parameters
```

Secrets live only in `.env` (git-ignored). Everything else is in
[configs/default.yaml](configs/default.yaml): models (`extractor` = Qwen3.8-27B with
reasoning off; `preannotator` / `diagnostic` / `reflection` = Claude Haiku 5.5 with
reasoning on), sampling params, provider pinning, budget, splits, metric weights, GEPA and
active-learning settings. `params` and `extra_body` are sent to OpenRouter unchanged. Any
key can be overridden from the command line:

```bash
uv run vex --set models.extractor.params.temperature=0.2 --set budget.max_usd=2 <command>
uv run vex --no-cache <command>          # bypass the LLM disk cache
uv run vex config show                   # resolved config + hash
```

Every command that calls a model writes `runs/<timestamp>-<command>-<hash>/` with the
resolved config, its hash, a log, its outputs and `cost.json`.

## Quick start

```bash
uv run vex prompt seed                         # seed candidate → prompts/seed.<hash>.json
uv run vex extract path/to/attestation.pdf     # one extraction, deterministic extractor
```

## Build the dataset (once)

```bash
uv run vex data fetch-specimens   # approved entries of data/specimen_sources.yaml → data/raw/real/
uv run vex data synth             # 200 seeded synthetic certificates → data/raw/synthetic/
uv run vex data register          # data/registry.jsonl (doc_id = sha256 of the file)
uv run vex data split             # frozen data/splits.yaml
uv run vex data sanity --n 20     # score a prompt against generator values (sanity only)
```

Raw files are git-ignored; the manifest, registry, splits and synthetic_meta are committed,
so `fetch-specimens` + `synth` rebuild the same files byte for byte. synthetic_meta is never
used as a label.

## The annotation-round workflow

```bash
# Round 0: held-out documents, annotated blind (no predictions shown)
uv run vex annotate push-blind
#   ... annotate ...
uv run vex annotate pull --round 0

# Rounds 1-3: bootstrap with micro-batches of 5, 10, 15 documents
uv run vex bootstrap --round 1    # pre-annotate (Haiku) + push
#   ... correct the pre-annotations ...
uv run vex bootstrap --round 1    # pull; when complete: proposed rewrite of the field
                                  # descriptions as a diff → confirm → prompts/bootstrap-r1.*.json
uv run vex report --round 1       # held-out score of the current prompt → reports/rounds.csv
uv run vex bootstrap --round 2    # starts from bootstrap-r1 ... and so on up to round 3

# Rounds 4+: optimize, then let active learning choose what to annotate next
uv run vex optimize               # GEPA on reviewed docs; held-out scored at the end
uv run vex select                 # uncertainty + diversity → round N+1 pushed pre-annotated
#   ... correct ...
uv run vex annotate pull --round 4
uv run vex report                 # stop when it reports a plateau
```

`vex optimize --max-metric-calls 100` sets a smaller budget, and `vex optimize --resume runs/<dir>`
continues an interrupted run. `vex eval --split gepa_pareto --candidate prompts/<file>.json`
scores any prompt on the reviewed documents of a GEPA bucket. Held-out splits are scored only
by `vex optimize` and `vex report`, and never fed back to any model that writes prompts.

Every annotation is appended to `data/annotations/round_<N>.jsonl`, the single source of
truth. Only `reviewed` and `blind` records count as gold; the latest one wins. Fields the
diagnostic model could not find in the document land in
`data/review_queue/suspect_annotations.jsonl`: check them, since a wrong label teaches the
prompt a wrong rule.

`reports/rounds.csv` columns: held-out score (global and per field, with the split used),
gold counts, share of fields annotators corrected, median review time, suspect
annotations, USD per round and cumulative, plateau flag.

## On this Mac vs at work

| | Mac (default) | Work |
|---|---|---|
| `annotation.backend` | `local`: editable JSON files in `data/inbox/round_<N>/` (fill `values`, write a `comment`, set `"_done": true`) | `kili`: set `KILI_API_KEY` / `KILI_API_ENDPOINT` in `.env`; the first run creates the project and logs its id, put it in `annotation.kili.project_id` |
| Held-out | `heldout_blind` is empty (only 1 public real specimen); the `carte_verte` layout is reserved as a stand-in (`splits.standin_layouts`), reported as `heldout_standin` | set `splits.standin_layouts: []`, add anonymized company documents to `data/raw/real/`, then `vex data register && vex data split` (existing assignments never move) |
| Provider | Parasail pinned for Qwen | consider adding `provider.data_collection: deny` (real personal data); check with `vex doctor` |

The Kili backend has **not** been run against a live instance (Kili is unreachable from this
Mac). Its payloads are validated offline with the SDK's own parser. Read
[docs/kili_uncertainties.md](docs/kili_uncertainties.md) before the first run.

## Tests

```bash
uv run pytest            # offline; all network calls are mocked
uv run pytest -m live    # one real extractor call through OpenRouter (< $0.01)
```

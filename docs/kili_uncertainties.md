# KiliBackend: what is verified and what is not

> **No code in this repository has run against a Kili instance.** Kili is not reachable
> from the development Mac. This page lists what was checked offline and what to check,
> in order, on the first run at work.

Kili SDK version: **2.142.1** (pinned, the company version, as in `~/Projects/template-kili`).

## Verified offline

| What | How |
| --- | --- |
| SDK method names and parameters used (`create_project`, `append_many_to_dataset`, `create_predictions`, `assets(metadata_where=…)`, `labels(asset_external_id_in=…, type_in=…, fields=…)`) | `inspect.signature` on the installed 2.142.1 SDK |
| `labels()` can return `labelOf.externalId`, `secondsToLabel`, `createdAt`, `labelType`, `author.email` | `kili/types.py` (`Label`, `LabelWithoutLabelOf`) in 2.142.1 |
| `json_interface` and prediction `jsonResponse` shapes | parsed by the SDK's own `ParsedJobs` (`tests/test_annotation.py`), the technique template-kili uses |
| Label → store record conversion, REVIEW-over-DEFAULT choice, blind/pre-annotated status from metadata | unit tests with canned labels |
| Push order and arguments (assets, then predictions only for non-blind items) | unit tests with a fake `Kili` client |
| Images are wrapped into one-page PDFs (project `input_type="PDF"`) | unit test + manual check on the Leocare JPEG |

## To check on the first run (in this order)

1. **Connection on-premise.** `KILI_API_ENDPOINT` must point to the company instance; without it the SDK targets Kili SaaS.
2. **Project creation.** With `annotation.kili.project_id: null`, the first `vex bootstrap` creates the project and logs its id. Put that id in your config so the next runs reuse it. Then open the project in the UI and check that the 10 transcription jobs plus COMMENTAIRE render as a form, and that the long French instructions display well (no known length limit, none verified).
3. **Local file upload.** `append_many_to_dataset` receives local PDF paths. Check that the SDK uploads them, and that the server doesn't expect a path it can read itself (template-kili uncertainty #6).
4. **`external_id` = full sha256 (64 hex chars).** No limit is documented. If Kili rejects it, shorten it in `kili_backend.py` and keep a mapping.
5. **Predictions show up as pre-annotations.** `create_predictions(model_name="vex-round<N>")` should prefill the transcription fields. Check that null fields really appear empty, and that `garanties` displays one guarantee per line.
6. **Metadata filter.** `pull` calls `assets(metadata_where={"round": N})`. Check that an integer metadata value filters as expected (Kili may store metadata as strings; if so, filter on `"N"`).
7. **Export shape.** Check that a reviewed label's `jsonResponse` is `{"JOB": {"text": "..."}}` and that a field left empty is either missing or `""`. Both are handled, but confirm it.
8. **Review time.** Check that `secondsToLabel` is filled on DEFAULT/REVIEW labels. If it is 0 or missing, `median review time` in `rounds.csv` will be empty for Kili rounds.
9. **Labels from the same annotator.** `latest_labels` keeps REVIEW over DEFAULT, then the most recent. Check that this matches the team's workflow, e.g. one labeller then one reviewer.
10. **Re-pull.** `pull` returns all labels of the round each time. The store skips records it already has (`append_new`), but a label edited after a pull becomes a new record. That is intended: the latest human record wins.

## Known constraints

- `kili==2.142.1` requires `pillow<10.1`, so the whole repository runs on Pillow 10.0.1. That was frozen before the synthetic data's `doc_id`s were computed (see PLAN §0).
- Kili 2.142.1 uses Pydantic v1's `pydantic.config.Extra`, deprecated in Pydantic v2 (a warning, filtered in pytest). It would break with Pydantic v3: keep Pydantic below 3 while Kili stays pinned.
- The SDK's transcription jobs also offer a `date` input widget. This repo follows the brief (free-text jobs everywhere) and normalizes French dates on import instead.

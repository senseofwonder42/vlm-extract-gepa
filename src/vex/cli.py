"""`vex` command-line interface.

Global options come before the command::

    uv run vex --set models.extractor.params.temperature=0.2 --no-cache extract doc.pdf
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, cast

import httpx
import typer
import yaml
from loguru import logger
from pydantic import ValidationError

from vex.annotation.backend import PushItem, make_backend
from vex.annotation.store import AnnotationStore, effective_split
from vex.bootstrap import (
    apply_rewrites,
    candidate_diff,
    collect_corrections,
    pre_annotate,
    propose_rewrites,
    select_batch,
)
from vex.data.registry import load_registry, register
from vex.data.sanity import eligible_documents, run_sanity
from vex.data.specimens import USER_AGENT, fetch_specimens
from vex.data.splits import Splits, assign_splits, load_splits, save_splits
from vex.data.synth.generator import generate
from vex.config import DEFAULT_CONFIG_PATH, Config, RoleName, Settings, config_hash, load_config
from vex.evaluate import score_candidate
from vex.extract import extract_document
from vex.llm.client import OpenRouterClient
from vex.llm.cost import CostTracker
from vex.metric.report import build_report
from vex.logs import setup_logging
from vex.prompt import Candidate, load_candidate, save_candidate, seed_candidate
from vex.runs import create_run_dir

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
config_app = typer.Typer(no_args_is_help=True, help="Inspect the configuration.")
prompt_app = typer.Typer(no_args_is_help=True, help="Manage candidate prompts.")
app.add_typer(config_app, name="config")
data_app = typer.Typer(no_args_is_help=True, help="Build the document dataset.")
annotate_app = typer.Typer(no_args_is_help=True, help="Exchange documents with annotators.")
app.add_typer(annotate_app, name="annotate")
app.add_typer(prompt_app, name="prompt")
app.add_typer(data_app, name="data")


class State:
    """Options shared by every command."""

    config: Config
    use_cache: bool


state = State()


@app.callback()
def main(
    config_path: Annotated[Path, typer.Option("--config", help="YAML config file.")] = DEFAULT_CONFIG_PATH,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Override a key: dotted.key=value (repeatable).")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the LLM disk cache for this run.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log at DEBUG level.")] = False,
) -> None:
    """Extract fields from French motor insurance certificates; optimize the prompt with GEPA."""
    setup_logging("DEBUG" if verbose else "INFO")
    try:
        state.config = load_config(config_path, overrides or [])
    except (ValidationError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config / --set") from exc
    state.use_cache = not no_cache


def _make_client(tracker: CostTracker) -> OpenRouterClient:
    key = Settings().openrouter_api_key
    if key is None or not key.get_secret_value():
        raise typer.BadParameter("OPENROUTER_API_KEY is missing: copy .env.example to .env and fill it in.")
    return OpenRouterClient(state.config, key, tracker, use_cache=state.use_cache)


@config_app.command("show")
def config_show() -> None:
    """Print the resolved configuration and its hash."""
    typer.echo(yaml.safe_dump(state.config.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    typer.echo(f"# config sha256: {config_hash(state.config)}")


@prompt_app.command("seed")
def prompt_seed(name: Annotated[str, typer.Option(help="File name prefix.")] = "seed") -> None:
    """Build the seed candidate from the schema and save it to `prompts/`."""
    path = save_candidate(seed_candidate(), state.config.paths.prompts, name)
    typer.echo(str(path))


@app.command()
def extract(
    document: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="PDF, PNG or JPEG file.")],
    candidate_path: Annotated[
        Path | None, typer.Option("--candidate", help="Saved candidate JSON (default: seed).")
    ] = None,
    role: Annotated[str, typer.Option(help="Role to use.")] = "extractor_eval",
) -> None:
    """Extract the fields of one document and print the JSON result."""
    if role not in ("extractor", "extractor_eval", "preannotator"):
        raise typer.BadParameter("role must be extractor, extractor_eval or preannotator")
    candidate = load_candidate(candidate_path) if candidate_path else seed_candidate()
    run_dir = create_run_dir(state.config, "extract")
    tracker = CostTracker(state.config.budget.max_usd)

    async def run() -> dict:
        async with _make_client(tracker) as client:
            result = await extract_document(
                client, document, candidate, role=cast(RoleName, role), use_cache=state.use_cache
            )
            return asdict(result)

    result = asyncio.run(run())
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "cost.json").write_text(json.dumps(tracker.summary(), indent=2), encoding="utf-8")
    if result["values"] is None:
        logger.error("Extraction failed: {}", result["error"])
    typer.echo(json.dumps(result["values"], indent=2, ensure_ascii=False))
    logger.info("Cost ${:.5f} ({} call(s)); path={} repaired={}", tracker.total_usd, len(tracker.calls),
                result["structured_path"], result["repaired"])


@app.command()
def doctor() -> None:
    """Check the API key and that each role's pinned providers support its parameters."""
    key = Settings().openrouter_api_key
    typer.echo(f"OPENROUTER_API_KEY: {'set' if key and key.get_secret_value() else 'MISSING'}")
    roles: list[RoleName] = ["extractor", "extractor_eval", "preannotator", "diagnostic", "reflection"]
    ok = True
    with httpx.Client(base_url=state.config.client.base_url, timeout=30) as http:
        for role in roles:
            role_config = state.config.role(role)
            endpoints = http.get(f"/models/{role_config.model}/endpoints").json().get("data", {}).get("endpoints", [])
            provider = role_config.extra_body.get("provider")
            order = provider.get("order") if isinstance(provider, dict) else None
            if order:
                endpoints = [e for e in endpoints if e.get("tag", "").split("/")[0] in order]
            wanted = {k for k in role_config.params if k != "max_tokens"} | {
                k for k in role_config.extra_body if k in ("reasoning",)
            }
            if not endpoints:
                ok = False
                typer.echo(f"[FAIL] {role}: no endpoint for {role_config.model} with providers {order}")
                continue
            structured = sum("structured_outputs" in e.get("supported_parameters", []) for e in endpoints)
            missing = sorted(set().union(*(wanted - set(e.get("supported_parameters", [])) for e in endpoints)))
            # Pinned: every pinned endpoint must accept every parameter. Unpinned:
            # require_parameters lets OpenRouter skip endpoints that lack one.
            failed = bool(missing) and bool(order)
            ok &= not failed
            typer.echo(
                f"[{'FAIL' if failed else 'ok'}] {role}: {role_config.model} via "
                f"{', '.join(e.get('tag', '?') for e in endpoints) if order else f'{len(endpoints)} endpoints'}; "
                f"structured_outputs on {structured}/{len(endpoints)}; unsupported somewhere: {missing or '-'}"
            )
    raise typer.Exit(0 if ok else 1)


@data_app.command("fetch-specimens")
def data_fetch_specimens(
    delay: Annotated[float, typer.Option(help="Seconds between two downloads.")] = 2.0,
) -> None:
    """Download approved real specimens (robots.txt respected) and update the manifest."""
    paths = state.config.paths
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        new = fetch_specimens(paths.specimen_sources, paths.raw / "real", paths.manifest, client, delay_s=delay)
    typer.echo(f"{len(new)} new specimen(s) downloaded")


@data_app.command("synth")
def data_synth(
    overwrite: Annotated[bool, typer.Option(help="Replace existing synthetic files.")] = False,
) -> None:
    """Generate the seeded synthetic certificates (`synth.n`, `synth.seed` in YAML)."""
    synth, paths = state.config.synth, state.config.paths
    try:
        files = generate(synth.n, synth.seed, paths.raw / "synthetic", paths.synthetic_meta,
                         scan_fraction=synth.scan_fraction, overwrite=overwrite)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"{len(files)} synthetic document(s) in {paths.raw / 'synthetic'}")


@data_app.command("register")
def data_register() -> None:
    """Add new files under data/raw/{real,synthetic} to the registry."""
    new = register(state.config.paths.raw, state.config.paths.registry)
    typer.echo(f"{len(new)} new document(s) registered")


@data_app.command("split")
def data_split() -> None:
    """Assign unassigned documents to splits; existing assignments never move."""
    paths = state.config.paths
    try:
        splits = assign_splits(load_registry(paths.registry), load_splits(paths.splits), state.config.splits)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    save_splits(splits, paths.splits)
    typer.echo(f"heldout_blind={len(splits.heldout_blind)} heldout_standin={len(splits.heldout_standin)} "
               f"reserved_unused={len(splits.reserved_unused)} gepa_train={len(splits.gepa_train)} "
               f"gepa_pareto={len(splits.gepa_pareto)} -> {paths.splits}")


@data_app.command("sanity")
def data_sanity(
    n: Annotated[int, typer.Option(help="Number of synthetic documents to score.")] = 20,
    candidate_path: Annotated[
        Path | None, typer.Option("--candidate", help="Saved candidate JSON (default: seed).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
) -> None:
    """Score a candidate on synthetic docs against generator values (sanity only, never labels)."""
    paths = state.config.paths
    splits = load_splits(paths.splits)
    if splits is None:
        raise typer.BadParameter("No splits yet: run `vex data register` and `vex data split` first.")
    documents = eligible_documents(load_registry(paths.registry), splits)
    candidate = load_candidate(candidate_path) if candidate_path else seed_candidate()
    run_dir = create_run_dir(state.config, "sanity")
    tracker = CostTracker(state.config.budget.max_usd)

    async def run() -> list:
        async with _make_client(tracker) as client:
            return await run_sanity(client, candidate, documents, paths.synthetic_meta, n, seed)

    scores = asyncio.run(run())
    report = build_report(scores)
    (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "documents.json").write_text(
        json.dumps([asdict(s) for s in scores], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "cost.json").write_text(json.dumps(tracker.summary(), indent=2), encoding="utf-8")
    typer.echo(report.table())
    typer.echo(f"cost ${tracker.total_usd:.4f} — details in {run_dir}")


def _load_splits_or_fail() -> Splits:
    splits = load_splits(state.config.paths.splits)
    if splits is None:
        raise typer.BadParameter("No splits yet: run `vex data register` and `vex data split` first.")
    return splits


def _previous_candidate(round_number: int) -> Candidate:
    """Latest `bootstrap-r<N-1>` candidate in prompts/, else the seed."""
    previous = sorted(state.config.paths.prompts.glob(f"bootstrap-r{round_number - 1}.*.json"),
                      key=lambda p: p.stat().st_mtime)
    if previous:
        typer.echo(f"Using candidate {previous[-1]}")
        return load_candidate(previous[-1])
    typer.echo("Using the seed candidate")
    return seed_candidate()


@app.command()
def bootstrap(
    round_number: Annotated[int, typer.Option("--round", help="Bootstrap round (1, 2, 3...).")],
    candidate_path: Annotated[
        Path | None, typer.Option("--candidate", help="Candidate for pre-annotation (default: previous round's).")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Apply the proposed rewrite without asking.")] = False,
) -> None:
    """Run one bootstrap round; call again after annotating to pull and rewrite descriptions."""
    config, paths = state.config, state.config.paths
    sizes = config.bootstrap.batch_sizes
    if not 1 <= round_number <= len(sizes):
        raise typer.BadParameter(f"--round must be between 1 and {len(sizes)} (bootstrap.batch_sizes)")
    splits, store = _load_splits_or_fail(), AnnotationStore(paths.annotations)
    backend = make_backend(config, Settings())
    candidate = load_candidate(candidate_path) if candidate_path else _previous_candidate(round_number)
    tracker = CostTracker(config.budget.max_usd)

    proposals = [r for r in store.round_records(round_number) if r.status == "pre_annotated"]
    if not proposals:
        batch = select_batch(round_number, sizes[round_number - 1], load_registry(paths.registry), splits,
                             store, config.splits.seed)
        if not batch:
            raise typer.BadParameter("No unannotated document left in the GEPA buckets.")
        create_run_dir(config, f"bootstrap-r{round_number}-push")

        async def run_push():
            async with _make_client(tracker) as client:
                return await pre_annotate(client, batch, candidate, round_number, config.annotation.backend)

        records, items = asyncio.run(run_push())
        store.append(records)
        backend.push(items, round_number)
        typer.echo(f"Round {round_number}: {len(items)} document(s) pre-annotated and pushed "
                   f"(${tracker.total_usd:.4f}). Correct them, then run this command again.")
        return

    pulled = store.append_new(backend.pull(round_number))
    reviewed = {r.doc_id for r in store.round_records(round_number) if r.status == "reviewed"}
    missing = [r.doc_id for r in proposals if r.doc_id not in reviewed]
    typer.echo(f"Round {round_number}: {len(pulled)} new review(s); {len(reviewed)}/{len(proposals)} done.")
    if missing:
        typer.echo("Finish the remaining documents, then run this command again.")
        return

    corrections = collect_corrections(store, round_number, splits, config.metric)
    if not corrections:
        typer.echo("Annotators changed nothing: no rewrite to propose.")
        return
    create_run_dir(config, f"bootstrap-r{round_number}-rewrite")

    async def run_rewrite():
        async with _make_client(tracker) as client:
            return await propose_rewrites(client, candidate, corrections)

    rewrites = asyncio.run(run_rewrite())
    if not rewrites:
        typer.echo(f"{len(corrections)} correction(s), but the reflection model proposed no change.")
        return
    proposed = apply_rewrites(candidate, rewrites)
    typer.echo(candidate_diff(candidate, proposed))
    for rewrite in rewrites:
        typer.echo(f"- {rewrite.field}: {rewrite.rationale}")
    if yes or typer.confirm("Apply this rewrite?", default=False):
        path = save_candidate(proposed, paths.prompts, f"bootstrap-r{round_number}")
        typer.echo(f"Saved {path}")
    else:
        typer.echo("Not applied.")


@annotate_app.command("push-blind")
def annotate_push_blind() -> None:
    """Push held-out documents (heldout_blind + heldout_standin) with NO predictions (round 0)."""
    paths = state.config.paths
    splits, store = _load_splits_or_fail(), AnnotationStore(paths.annotations)
    registry = {r.doc_id: r for r in load_registry(paths.registry)}
    done = store.annotated_doc_ids()
    todo = [d for d in [*splits.heldout_blind, *splits.heldout_standin] if d not in done]
    items = [PushItem(d, Path(registry[d].path), registry[d].origin, "blind", None) for d in todo]
    make_backend(state.config, Settings()).push(items, 0)
    typer.echo(f"{len(items)} held-out document(s) pushed blind (round 0).")


@annotate_app.command("pull")
def annotate_pull(round_number: Annotated[int, typer.Option("--round", help="Round to pull (0 = blind).")]) -> None:
    """Import finished annotations of a round into the store."""
    store = AnnotationStore(state.config.paths.annotations)
    new = store.append_new(make_backend(state.config, Settings()).pull(round_number))
    typer.echo(f"{len(new)} new record(s) stored for round {round_number}.")


@app.command("eval")
def eval_command(
    split: Annotated[str, typer.Option(help="gepa_train or gepa_pareto (held-out splits are refused).")],
    candidate_path: Annotated[
        Path | None, typer.Option("--candidate", help="Saved candidate JSON (default: seed).")
    ] = None,
) -> None:
    """Score a candidate on the gold documents of a GEPA bucket."""
    if split not in ("gepa_train", "gepa_pareto"):
        raise typer.BadParameter("only gepa_train or gepa_pareto; held-out splits are scored by optimize/report")
    paths = state.config.paths
    store = AnnotationStore(paths.annotations)
    doc_ids = effective_split(cast(Literal["gepa_train", "gepa_pareto"], split), _load_splits_or_fail(), store)
    if not doc_ids:
        raise typer.BadParameter(f"No gold document in {split} yet: run bootstrap rounds first.")
    candidate = load_candidate(candidate_path) if candidate_path else seed_candidate()
    run_dir = create_run_dir(state.config, f"eval-{split}")
    tracker = CostTracker(state.config.budget.max_usd)
    registry = {r.doc_id: r for r in load_registry(paths.registry)}

    async def run():
        async with _make_client(tracker) as client:
            return await score_candidate(client, candidate, doc_ids, registry, store.gold())

    scores = asyncio.run(run())
    report = build_report(scores)
    (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "documents.json").write_text(json.dumps([asdict(s) for s in scores], indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    (run_dir / "cost.json").write_text(json.dumps(tracker.summary(), indent=2), encoding="utf-8")
    typer.echo(report.table())
    typer.echo(f"cost ${tracker.total_usd:.4f} — details in {run_dir}")

"""`vex optimize`: run GEPA on the reviewed documents, then score on held-out.

Data mapping (docs/gepa_guide.md explains why):
- GEPA `trainset` = `gepa_train` bucket ∩ gold → minibatches for reflection;
- GEPA `valset`   = `gepa_pareto` bucket ∩ gold → Pareto scoring of every
  accepted candidate. It drives selection, so it is NOT a held-out set;
- `heldout_blind` (real) and `heldout_standin` (Mac stand-in) are touched
  ONLY by `score_heldout`, after GEPA has finished, and by `vex report`.

Run directory: `config.resolved.yaml`, `log.txt`, `gepa/` (GEPA's own state,
resumable), `iterations.jsonl`, `candidates.jsonl`, `proposals.jsonl`,
`candidates/<idx>.json`, `pareto_front.json`, `result.json`, `heldout.json`,
`cost.json`. The best candidate is also saved to `prompts/gepa-best.<hash>.json`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import gepa
from loguru import logger

from vex.annotation.store import AnnotationRecord, AnnotationStore, effective_split, held_out_ids
from vex.config import Config, Settings
from vex.data.registry import DocRecord, load_registry
from vex.data.splits import Splits, load_splits
from vex.diagnostic import Diagnostician
from vex.evaluate import score_candidate
from vex.gepa_adapter import DocInstance, ErrorDrivenSelector, VexAdapter
from vex.llm.client import OpenRouterClient
from vex.llm.cost import BudgetExceeded, CostTracker
from vex.llm.sync import BackgroundLoop
from vex.metric.report import build_report
from vex.prompt import Candidate, candidate_hash, save_candidate

HELDOUT_SPLITS = ("heldout_blind", "heldout_standin")


class ReflectionLM:
    """GEPA `reflection_lm`: a sync callable `(prompt | messages) -> str`.

    GEPA's default proposer renders the reflective dataset into a prompt and
    calls this object. Because it exposes `total_cost`, GEPA uses it as-is
    (no litellm wrapper), and `max_reflection_cost` reads our real spend.
    """

    def __init__(self, client: OpenRouterClient, loop: BackgroundLoop) -> None:
        """Create the wrapper.

        Args:
            client: OpenRouter client (uses the `reflection` role).
            loop: Background loop the client lives on.
        """
        self.client = client
        self.loop = loop

    def __call__(self, prompt: str | list[dict]) -> str:
        """Send one reflection prompt and return the reply text."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.loop.run(self.client.complete("reflection", messages)).content

    @property
    def total_cost(self) -> float:
        """USD spent by the reflection role so far."""
        return sum(call.usd for call in self.client.tracker.calls if call.role == "reflection")


class BudgetStopper:
    """GEPA stop callback: stop between iterations once `budget.max_usd` is spent."""

    def __init__(self, tracker: CostTracker) -> None:
        """Create the stopper.

        Args:
            tracker: The run's cost tracker.
        """
        self.tracker = tracker

    def __call__(self, gepa_state: object) -> bool:
        """Return True when the budget is spent."""
        return self.tracker.total_usd >= self.tracker.max_usd


class RunRecorder:
    """GEPA callback writing per-iteration logs into the run directory."""

    def __init__(self, run_dir: Path, tracker: CostTracker) -> None:
        """Create the recorder.

        Args:
            run_dir: Run directory.
            tracker: Cost tracker (USD is logged with each event).
        """
        self.run_dir = run_dir
        self.tracker = tracker

    def _write(self, name: str, entry: dict) -> None:
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**entry, "usd_so_far": round(self.tracker.total_usd, 6)}, ensure_ascii=False,
                                    default=str) + "\n")

    def on_valset_evaluated(self, event: dict) -> None:
        """A candidate was scored on the whole valset (`gepa_pareto`)."""
        self._write("candidates.jsonl", {
            "iteration": event["iteration"], "candidate_idx": event["candidate_idx"],
            "average_score": event["average_score"], "parent_ids": list(event["parent_ids"]),
            "candidate_hash": candidate_hash(event["candidate"]), "candidate": event["candidate"],
        })

    def on_proposal_end(self, event: dict) -> None:
        """The reflection model proposed new component texts."""
        self._write("proposals.jsonl", {"iteration": event["iteration"], "new_texts": event["new_instructions"]})

    def on_candidate_accepted(self, event: dict) -> None:
        """A rewrite beat its parent on the minibatch."""
        self._write("iterations.jsonl", {"iteration": event["iteration"], "event": "accepted",
                                         "new_candidate_idx": event["new_candidate_idx"],
                                         "minibatch_score": event["new_score"]})

    def on_candidate_rejected(self, event: dict) -> None:
        """A rewrite did not beat its parent on the minibatch."""
        self._write("iterations.jsonl", {"iteration": event["iteration"], "event": "rejected",
                                         "old_score": event["old_score"], "new_score": event["new_score"],
                                         "reason": event["reason"]})

    def on_iteration_end(self, event: dict) -> None:
        """End of an iteration: record the metric calls used so far."""
        self._write("iterations.jsonl", {"iteration": event["iteration"], "event": "end",
                                         "proposal_accepted": event["proposal_accepted"],
                                         "total_metric_calls": event["state"].total_num_evals})


@dataclass(frozen=True)
class OptimizeOutcome:
    """What `vex optimize` produced.

    Attributes:
        best: Best candidate on the valset.
        best_path: Where it was saved in `prompts/`.
        heldout: Held-out scores per split and per scored candidate.
        total_usd: Total spend of the run.
        stopped_by_budget: True if the budget guard interrupted GEPA.
    """

    best: Candidate
    best_path: Path
    heldout: dict
    total_usd: float
    stopped_by_budget: bool


def _instances(doc_ids: list[str], registry: dict[str, DocRecord], gold: dict[str, AnnotationRecord]) -> list[DocInstance]:
    return [DocInstance(doc_id=d, path=Path(registry[d].path), gold=gold[d].values) for d in doc_ids]


def score_heldout(
    client: OpenRouterClient, loop: BackgroundLoop, candidates: dict[str, Candidate], splits: Splits,
    store: AnnotationStore, registry: dict[str, DocRecord],
) -> dict:
    """Score candidates on the held-out splits. The only reader of held-out gold.

    Results are reported, never fed back to GEPA, the reflection model or the
    bootstrap rewriter. Use them to decide when to stop, not to choose between
    candidates (docs/gepa_guide.md).

    Args:
        client: OpenRouter client.
        loop: Background loop the client lives on.
        candidates: Name → candidate (e.g. seed, best, pareto-2).
        splits: Frozen splits.
        store: Annotation store.
        registry: doc_id → registry record.

    Returns:
        `{split: {"n": int, "scores": {name: report dict}}}`; a split without
        gold documents has `n = 0` and no scores.
    """
    gold = store.gold()
    results: dict = {}
    for split in HELDOUT_SPLITS:
        doc_ids = [d for d in getattr(splits, split) if d in gold]
        results[split] = {"n": len(doc_ids), "scores": {}}
        if not doc_ids:
            logger.warning("{}: no gold document yet (annotate it blind first); not scored", split)
            continue
        for name, candidate in candidates.items():
            scores = loop.run(score_candidate(client, candidate, doc_ids, registry, gold))
            results[split]["scores"][name] = build_report(scores).to_dict()
            logger.info("{} on {} ({} docs): {:.3f}", name, split, len(doc_ids), results[split]["scores"][name]["global_score"])
    return results


def _best_from_recorder(run_dir: Path, seed: Candidate) -> tuple[Candidate, list[tuple[float, Candidate]]]:
    path = run_dir / "candidates.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    ranked = sorted(((e["average_score"], e["candidate"]) for e in entries), key=lambda x: -x[0])
    return (ranked[0][1] if ranked else seed), ranked


def run_optimize(config: Config, settings: Settings, seed: Candidate, run_dir: Path, use_cache: bool = True) -> OptimizeOutcome:
    """Run GEPA, save its outputs, then score the result on held-out documents.

    Args:
        config: Resolved configuration.
        settings: Secrets.
        seed: Seed candidate.
        run_dir: Run directory (created by `create_run_dir`).
        use_cache: LLM disk cache switch.

    Returns:
        The outcome.

    Raises:
        ValueError: If `gepa_train` or `gepa_pareto` has no gold document yet.
        RuntimeError: If a held-out document reached the GEPA data (guard).
    """
    paths = config.paths
    splits = load_splits(paths.splits)
    if splits is None:
        raise ValueError("No splits: run `vex data split` first")
    store = AnnotationStore(paths.annotations)
    registry = {r.doc_id: r for r in load_registry(paths.registry)}
    gold = store.gold()
    train_ids = effective_split("gepa_train", splits, store)
    val_ids = effective_split("gepa_pareto", splits, store)
    if not train_ids or not val_ids:
        raise ValueError(f"Need gold documents in both buckets (train={len(train_ids)}, pareto={len(val_ids)})")
    if leaked := (set(train_ids) | set(val_ids)) & held_out_ids(splits):
        raise RuntimeError(f"Held-out documents in GEPA data: {sorted(leaked)[:3]}")
    trainset, valset = _instances(train_ids, registry, gold), _instances(val_ids, registry, gold)
    logger.info("GEPA trainset={} valset={} max_metric_calls={}", len(trainset), len(valset), config.gepa.max_metric_calls)

    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is missing from .env")
    # Budget: GEPA may spend up to max_usd - heldout_reserve_usd; held-out scoring
    # afterwards may use the reserve. Both phases are enforced by the same tracker.
    tracker = CostTracker(config.budget.max_usd - config.budget.heldout_reserve_usd)
    loop = BackgroundLoop()
    client = OpenRouterClient(config, settings.openrouter_api_key, tracker, use_cache=use_cache)
    adapter = VexAdapter(config, client, loop, Diagnostician(client, config.cache.dir.parent / "diagnostics",
                                                             paths.review_queue))
    g = config.gepa
    module_selector = ErrorDrivenSelector(config.metric.weights) if g.module_selector == "error_driven" else g.module_selector
    stopped_by_budget = False
    result = None
    try:
        result = gepa.optimize(
            seed_candidate=seed,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=ReflectionLM(client, loop),
            candidate_selection_strategy=g.candidate_selection_strategy,
            skip_perfect_score=g.skip_perfect_score,
            perfect_score=1.0,
            reflection_minibatch_size=min(g.reflection_minibatch_size, len(trainset)),
            module_selector=module_selector,
            use_merge=g.use_merge,
            max_merge_invocations=g.max_merge_invocations,
            max_metric_calls=g.max_metric_calls,
            max_reflection_cost=g.max_reflection_cost_usd,
            stop_callbacks=[BudgetStopper(tracker)],
            run_dir=str(run_dir / "gepa"),
            callbacks=[RunRecorder(run_dir, tracker)],
            seed=g.seed,
            cache_evaluation=g.cache_evaluation,
            display_progress_bar=False,
        )
    except BudgetExceeded as exc:
        stopped_by_budget = True
        logger.warning("Budget guard stopped GEPA mid-iteration ({}); using the best candidate recorded so far", exc)

    if result is not None:
        best = result.best_candidate
        ranked = sorted(range(result.num_candidates), key=lambda i: -result.val_aggregate_scores[i])
        top = [(result.val_aggregate_scores[i], result.candidates[i]) for i in ranked]
        candidates_dir = run_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)
        for index, candidate in enumerate(result.candidates):
            (candidates_dir / f"{index}.json").write_text(json.dumps({
                "index": index, "parents": result.parents[index], "val_score": result.val_aggregate_scores[index],
                "hash": candidate_hash(candidate), "candidate": candidate}, ensure_ascii=False, indent=2), "utf-8")
        front = {str(k): sorted(v) for k, v in result.per_val_instance_best_candidates.items()}
        (run_dir / "pareto_front.json").write_text(json.dumps(
            {"per_val_instance_best_candidates": front, "val_ids": val_ids}, indent=2), "utf-8")
        (run_dir / "result.json").write_text(json.dumps(result.to_dict(), default=str, indent=2), "utf-8")
    else:
        best, top = _best_from_recorder(run_dir, seed)

    best_path = save_candidate(best, paths.prompts, "gepa-best")
    tracker.max_usd = config.budget.max_usd  # held-out phase: the reserve becomes available
    to_score = {"seed": seed, "best": best}
    for rank, (_, candidate) in enumerate(top[: g.heldout_top_k], start=1):
        if candidate_hash(candidate) not in {candidate_hash(c) for c in to_score.values()}:
            to_score[f"pareto-{rank}"] = candidate
    heldout = score_heldout(client, loop, to_score, splits, store, registry)
    (run_dir / "heldout.json").write_text(json.dumps(heldout, indent=2), "utf-8")
    (run_dir / "cost.json").write_text(json.dumps(tracker.summary(), indent=2), "utf-8")
    loop.run(client.aclose())
    loop.close()
    return OptimizeOutcome(best, best_path, heldout, tracker.total_usd, stopped_by_budget)

"""GEPA adapter: how GEPA runs, scores and learns from our extraction prompt.

Read docs/gepa_guide.md first if GEPA is new to you. In one paragraph:

GEPA evolves a *candidate*, the dict `{"instructions": ..., "field:<name>": ...}`
built in `vex.prompt`. Each key is a *component* GEPA can rewrite. To decide
what to rewrite and whether a rewrite helped, GEPA needs two things from us,
and this adapter provides exactly those two:

1. `evaluate(batch, candidate, capture_traces)`: run the extractor with this
   candidate on a batch of documents and return one score per document
   (+ "trajectories", our own record of what happened, when GEPA asks for them).
2. `make_reflective_dataset(candidate, eval_batch, components_to_update)`:
   turn those trajectories into short, readable feedback records for the
   components GEPA chose to rewrite. GEPA hands them to the reflection model,
   which writes a new text for each component.

GEPA never looks inside trajectories: their shape is ours to choose.

GEPA's API (verified in the installed gepa 0.1.4 source, `gepa/core/adapter.py`):
- `evaluate` must return `EvaluationBatch(outputs, scores, trajectories)` with
  `len(outputs) == len(scores) == len(batch)`, and trajectories only when
  `capture_traces=True`. Higher scores are better.
- It must NOT raise for a single bad document: a failed document scores 0 and
  its trajectory explains why. Exceptions are for systemic failures only.
- GEPA sums minibatch scores to accept or reject a rewrite, and averages valset
  scores to track candidates and build the Pareto front.
"""

import asyncio
import base64
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gepa import EvaluationBatch
from gepa.image import Image
from loguru import logger
from pydantic import JsonValue

from vex.config import Config
from vex.diagnostic import Diagnostician, FieldDiagnosis
from vex.extract import ExtractionResult, extract_document
from vex.llm.client import LLMError, OpenRouterClient
from vex.llm.media import load_pages
from vex.llm.sync import BackgroundLoop
from vex.metric.compare import DocumentScore, score_document
from vex.prompt import INSTRUCTIONS_KEY, Candidate, component_key
from vex.schema import FIELD_NAMES


@dataclass(frozen=True)
class DocInstance:
    """GEPA's `DataInst`: one document with its gold values.

    GEPA treats data instances as opaque; it only passes them back to
    `evaluate`. Only documents with a `reviewed` or `blind` gold record from
    the GEPA buckets are ever turned into a `DocInstance` (see `vex.optimize`).

    Attributes:
        doc_id: Document id.
        path: Document file.
        gold: Gold values (`AttestationAuto` fields).
    """

    doc_id: str
    path: Path
    gold: dict[str, JsonValue]


# Trajectory = a plain dict, one per document (our own format, GEPA does not read it):
#   {"doc_id", "path", "raw_outputs", "parsed", "structured_path", "repaired",
#    "error", "score", "fields": [{"field", "gold", "pred", "score", "error", "diff"}]}
Trajectory = dict
# RolloutOutput = the parsed values (or None when the reply could not be parsed).
RolloutOutput = dict[str, JsonValue] | None


class VexAdapter:
    """`GEPAAdapter` implementation for document field extraction."""

    # Extension point (out of scope here, see docs/gepa_guide.md): GEPA calls
    # `adapter.propose_new_texts(candidate, reflective_dataset, components)` when
    # this attribute is a callable, INSTEAD of its default text-only reflection.
    # A multimodal proposer would go here. Leaving it `None` makes GEPA use the
    # `reflection_lm` passed to `gepa.optimize`.
    propose_new_texts = None

    def __init__(self, config: Config, client: OpenRouterClient, loop: BackgroundLoop,
                 diagnostician: Diagnostician) -> None:
        """Create the adapter.

        Args:
            config: Resolved configuration.
            client: OpenRouter client. It must be used only on `loop`.
            loop: Background event loop (GEPA calls us synchronously).
            diagnostician: Diagnoses failing fields for the reflective dataset.
        """
        self.config = config
        self.client = client
        self.loop = loop
        self.diagnostician = diagnostician

    # ------------------------------------------------------------------ evaluate

    def evaluate(
        self, batch: list[DocInstance], candidate: Candidate, capture_traces: bool = False
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Run the extractor with `candidate` on `batch` and score each document.

        Called by GEPA (a) on a training minibatch with `capture_traces=True`,
        before proposing a rewrite; (b) on the same minibatch for the rewritten
        candidate, to accept or reject it; (c) on the whole valset (`gepa_pareto`)
        for accepted candidates. Each document scored = one "metric call" of
        the `max_metric_calls` budget.

        Args:
            batch: Documents to run.
            candidate: Candidate prompt (never mutated).
            capture_traces: Whether GEPA wants trajectories for reflection.

        Returns:
            Outputs (parsed values), scores in [0, 1], and trajectories if asked.
        """
        results = self.loop.run(self._extract_all(batch, candidate))
        outputs, scores, trajectories = [], [], []
        for instance, result in zip(batch, results, strict=True):
            if isinstance(result, Exception):
                # A transport or API failure on one document must not kill the
                # run (GEPA's contract): score 0 and record why. BudgetExceeded is
                # NOT caught here: it propagates and stops the run (see optimize.py).
                logger.error("{}: extraction failed ({})", instance.doc_id[:12], result)
                doc_score = score_document(instance.doc_id, instance.gold, None, self.config.metric)
                trajectory = self._trajectory(instance, None, doc_score, error=str(result))
                outputs.append(None)
            else:
                # The P3 metric: weighted field scores; a parse failure scores 0.
                doc_score = score_document(instance.doc_id, instance.gold, result.values, self.config.metric)
                trajectory = self._trajectory(instance, result, doc_score, error=result.error)
                outputs.append(result.values)
            scores.append(doc_score.score)
            trajectories.append(trajectory)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    async def _extract_all(self, batch: list[DocInstance], candidate: Candidate) -> list:
        # `extractor_eval` = the deterministic extractor (temperature 0, §9 Q2), so
        # a score change comes from the prompt, not from sampling luck.
        # LLMError (a call that failed for good) is returned as a value, not raised;
        # BudgetExceeded is let through.
        async def one(instance: DocInstance) -> ExtractionResult | LLMError:
            try:
                return await extract_document(self.client, instance.path, candidate, role="extractor_eval")
            except LLMError as exc:
                return exc

        return await asyncio.gather(*(one(instance) for instance in batch))

    @staticmethod
    def _trajectory(instance: DocInstance, result: ExtractionResult | None, doc_score: DocumentScore,
                    error: str | None) -> Trajectory:
        return {
            "doc_id": instance.doc_id,
            "path": str(instance.path),
            "raw_outputs": result.raw_outputs if result else [],
            "parsed": result.values if result else None,
            "structured_path": result.structured_path if result else None,
            "repaired": result.repaired if result else False,
            "error": error,
            "score": doc_score.score,
            "fields": [
                {"field": f.field, "gold": f.gold, "pred": f.pred, "score": f.score, "error": f.error, "diff": f.diff}
                for f in doc_score.fields
            ],
        }

    # -------------------------------------------------- make_reflective_dataset

    def make_reflective_dataset(
        self, candidate: Candidate, eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Build the feedback the reflection model reads, per component to rewrite.

        Steps:
        1. collect every failing field of every trajectory in the minibatch;
        2. for failing fields whose component GEPA wants to update, ask the
           `diagnostic` model to locate the gold value in the document and quote
           it (cached; unlocated cases go to the suspect-annotation queue and
           are excluded; diagnoses without a quote are dropped);
        3. route each record to its `field:<name>` component, and give
           `instructions` a per-document summary (parse failures, repairs,
           which fields failed).

        Records follow GEPA's recommended shape `{"Inputs", "Generated Outputs",
        "Feedback"}`. GEPA renders them as markdown in the reflection prompt.

        Args:
            candidate: Candidate that produced `eval_batch`.
            eval_batch: Result of `evaluate(..., capture_traces=True)`.
            components_to_update: Components GEPA chose to rewrite this iteration.

        Returns:
            `{component: [record, ...]}` for each requested component.
        """
        trajectories = eval_batch.trajectories or []
        wanted_fields = {name for name in FIELD_NAMES if component_key(name) in components_to_update}

        # 1-2. Failing fields → diagnoses (only for the components being rewritten: saves calls).
        failing = [
            (traj, field)
            for traj in trajectories
            for field in traj["fields"]
            if field["error"] != "correct" and field["field"] in wanted_fields and traj["parsed"] is not None
        ]
        diagnoses = self.loop.run(self._diagnose_all(candidate, failing))

        # 3. Routing.
        dataset: dict[str, list[dict]] = defaultdict(list)
        for (traj, field), diagnosis in zip(failing, diagnoses, strict=True):
            if diagnosis is None:
                continue  # dropped (no quote) or suspect annotation: never taught to the prompt
            record = {
                "Inputs": {"field": field["field"], "document": Path(traj["path"]).name},
                "Generated Outputs": field["pred"],
                "Feedback": (
                    f"{field['diff']}\n"
                    f"Dans le document / In the document — {diagnosis.location} : « {diagnosis.quote} »\n"
                    f"Cause : {diagnosis.explanation}"
                ),
            }
            if self.config.gepa.reflection_images:
                # Off by default (§9 Q8). GEPA's default proposer sends gepa.Image
                # objects found in records to the reflection LM as image parts.
                record["Document image"] = self._first_page(Path(traj["path"]))
            dataset[component_key(field["field"])].append(record)

        if INSTRUCTIONS_KEY in components_to_update:
            for traj in trajectories:
                dataset[INSTRUCTIONS_KEY].append(self._instructions_record(traj))

        # A requested component with no usable record (e.g. round_robin picked a
        # field that was correct everywhere) still gets a record, so the
        # reflection model knows nothing is broken rather than seeing nothing.
        for component in components_to_update:
            if not dataset.get(component):
                dataset[component] = [{
                    "Inputs": {"component": component},
                    "Generated Outputs": "no error on this minibatch",
                    "Feedback": "Aucune erreur sur ce lot pour ce champ / No error on this minibatch: "
                                "keep the description, only clarify it if clearly useful.",
                }]
        logger.info("Reflective dataset: {}", {k: len(v) for k, v in dataset.items()})
        return {component: dataset[component] for component in components_to_update}

    async def _diagnose_all(self, candidate: Candidate, failing: list[tuple[Trajectory, dict]]) -> list[FieldDiagnosis | None]:
        return await asyncio.gather(*(
            self.diagnostician.diagnose(traj["doc_id"], Path(traj["path"]), field["field"],
                                        candidate[component_key(field["field"])], field["pred"], field["gold"])
            for traj, field in failing
        ))

    @staticmethod
    def _instructions_record(traj: Trajectory) -> dict:
        failed = [f for f in traj["fields"] if f["error"] != "correct"]
        if traj["parsed"] is None:
            feedback = (f"La réponse n'a pas pu être lue comme JSON valide (après une relance) / "
                        f"Unparseable output: {traj['error']}. Tous les champs comptent comme oubliés.")
        elif not failed:
            feedback = "Extraction parfaite / Perfect extraction."
        else:
            feedback = "\n".join(f["diff"] for f in failed)
        if traj["repaired"]:
            feedback += "\nNote : la première réponse était invalide et a dû être réparée / first reply needed a repair."
        return {
            "Inputs": {"document": Path(traj["path"]).name},
            "Generated Outputs": traj["parsed"],
            "Feedback": f"Score {traj['score']:.2f}. {feedback}",
        }

    def _first_page(self, path: Path) -> Image:
        page = load_pages(path, dpi=self.config.render.dpi, max_pages=1)[0]
        return Image(base64_data=base64.b64encode(page.data).decode("ascii"), media_type=page.mime)


class ErrorDrivenSelector:
    """GEPA `module_selector`: rewrite the component that is failing most.

    GEPA calls it with the parent's minibatch trajectories and scores, and
    expects the list of component names to rewrite this iteration. The default
    `round_robin` cycles through all 11 components even when a field is
    perfect, which wastes 2·minibatch metric calls testing a blind rewrite.
    This selector picks:
    - `instructions` if any document's output could not be parsed (a global problem);
    - otherwise the `field:<name>` with the largest weighted error
      Σ weight·(1 − field score) over the minibatch (ties: schema order);
    - `instructions` if nothing failed (GEPA skips perfect minibatches anyway).
    """

    def __init__(self, weights: dict[str, float]) -> None:
        """Create the selector.

        Args:
            weights: Field weights (`metric.weights`).
        """
        self.weights = weights

    def __call__(self, state: object, trajectories: list[Trajectory], subsample_scores: list[float],
                 candidate_idx: int, candidate: Candidate) -> list[str]:
        """Return the single component to rewrite (see class docstring)."""
        if any(traj["parsed"] is None for traj in trajectories):
            return [INSTRUCTIONS_KEY]
        error = dict.fromkeys(FIELD_NAMES, 0.0)
        for traj in trajectories:
            for field in traj["fields"]:
                error[field["field"]] += self.weights[field["field"]] * (1.0 - field["score"])
        worst = max(FIELD_NAMES, key=lambda name: error[name])  # max() keeps the first on ties
        return [component_key(worst)] if error[worst] > 0 else [INSTRUCTIONS_KEY]

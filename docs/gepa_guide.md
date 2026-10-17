# GEPA for this project: a beginner's guide

GEPA ("Genetic-Pareto", the `gepa` library, version **0.1.4** here) improves the *texts* of
a system (here, our extraction prompt) by running it, reading what went wrong, and asking
a language model to rewrite the texts. It keeps several good versions instead of a single
"best" one, and only trusts a rewrite once it has scored it.

Everything below was checked against the installed `gepa` source (`gepa/api.py`,
`gepa/core/adapter.py`, `gepa/core/engine.py`, `gepa/proposer/reflective_mutation/`).
For a live, inspectable run, open
[notebooks/gepa_walkthrough.ipynb](../notebooks/gepa_walkthrough.ipynb).

---

## 1. Candidate and components

A **candidate** is a `dict[str, str]`. Each key is a **component**: a text GEPA may rewrite
on its own. Ours has 11 components (`vex.prompt.seed_candidate`):

```python
{
    "instructions":        "Tu extrais des champs d'un justificatif ...",   # global rules
    "field:assureur":      "Nom de la compagnie d'assurance ...",          # one per schema field
    ...
    "field:garanties":     "Liste des garanties souscrites ...",
}
```

`vex.prompt.render(candidate)` turns it into what the extractor actually receives: a system
prompt (instructions + field list) and a strict JSON schema whose field descriptions come
from the candidate. Candidates are saved as `prompts/<name>.<hash8>.json`.

Splitting the prompt per field lets GEPA fix one field's rules without touching the others,
and makes every change readable as a diff.

## 2. One GEPA iteration

```
           ┌──────────────────────── candidate pool (Pareto front) ───────────────────────┐
           │                                                                               │
 1. select a parent ──► 2. evaluate it on a training MINIBATCH (with traces)                │
                              │                                                            │
                              ▼                                                            │
                        3. choose component(s) to rewrite   (module_selector)              │
                              │                                                            │
                              ▼                                                            │
                        4. make_reflective_dataset          (our feedback records)         │
                              │                                                            │
                              ▼                                                            │
                        5. reflection LM writes new text(s) (GEPA's default proposer)      │
                              │                                                            │
                              ▼                                                            │
                        6. evaluate the child on the SAME minibatch                        │
                              │ sum(child) > sum(parent) ?                                 │
                       no ◄───┴───► yes: 7. evaluate the child on the full VALSET ─────────┘
                    (rejected)            and add it to the pool
```

In detail:

1. **Select a parent.** With `candidate_selection_strategy: pareto`, GEPA samples among
   candidates that are best on at least one valset document (the *Pareto front*), so a
   candidate that is the only one to handle green cards is kept even if its average is lower.
2. **Evaluate on a minibatch** of `reflection_minibatch_size` training documents, with
   `capture_traces=True`. Our `VexAdapter.evaluate` runs the extractor, scores each document
   with the P3 metric, and returns the scores plus our *trajectories* (raw output, parsed
   values, per-field diff). If every score is perfect, GEPA skips the rest of the iteration
   (`skip_perfect_score`), but the evaluation was already paid for.
3. **Choose what to rewrite.** GEPA calls the `module_selector`. Ours,
   `ErrorDrivenSelector`, picks the field with the largest weighted error on the minibatch
   (or `instructions` if an output could not be parsed). GEPA's default `round_robin` would
   cycle through all 11 components, including correct ones.
4. **Build the reflective dataset.** `VexAdapter.make_reflective_dataset` turns the
   trajectories into short records `{"Inputs", "Generated Outputs", "Feedback"}` for the
   chosen component. For each failing field, the `diagnostic` model looks at the page and
   must quote where the expected value is. Unquoted diagnoses are dropped; an expected
   value it cannot find goes to `data/review_queue/suspect_annotations.jsonl` (maybe a
   labelling error) and is never taught to the prompt.
5. **Reflect.** GEPA's default proposer (`InstructionProposalSignature`) puts the current
   text and the records into a fixed prompt ("I provided an assistant with the following
   instructions...") and calls our `reflection_lm` (Haiku). The new text is read from the
   reply's ``` block.
6. **Acceptance test.** The child (parent + new text) runs on the *same* minibatch. It is
   accepted only if the **sum** of its scores is strictly higher (`strict_improvement`).
7. **Valset evaluation.** An accepted child is scored on every valset document. Its scores
   update the Pareto front, and the candidate with the best *average* becomes `best_candidate`.

**Merge** (`use_merge: true`, `gepa/proposer/merge.py`): after an iteration that produced a
new candidate, GEPA may try a merge instead of a reflective mutation. It takes two
candidates with a common ancestor and starts from a copy of that ancestor. For each
component, if only one parent changed it, that parent's text is used; if both changed it,
the text comes from the parent with the better valset average (random on a tie). The merged
candidate is scored on a small subsample of valset documents both parents were scored on,
and kept only if it does at least as well as both. It needs evaluations but no reflection
call. At most `max_merge_invocations` merges per run.

## 3. Our datasets: GEPA's "valset" is not a held-out set

| Our split | GEPA argument | Used for | Held out? |
|---|---|---|---|
| `gepa_train` ∩ gold | `trainset` | minibatches: reflection + acceptance test | no |
| `gepa_pareto` ∩ gold | `valset` | scoring accepted candidates, Pareto front, `best_candidate` | **no**: it drives selection |
| `heldout_blind` (real docs) | never passed to GEPA | final score of the run, `vex report` | **yes** |
| `heldout_standin` (Mac only, reserved synthetic layout) | never passed to GEPA | same, reported separately | **yes** |

GEPA picks candidates by their valset scores, so those scores are optimistic. **Your
"validation" is `heldout_blind`**: annotated blind, never shown to GEPA, the diagnostic
model, the reflection model or the bootstrap rewriter (tests guard each entry point). It is
scored only after GEPA finishes (`vex.optimize.score_heldout`) and by `vex report`. On the
Mac, where too few real specimens exist, `heldout_standin` plays that role: it measures
generalization to an unseen layout, not real-world accuracy.

Use held-out scores to decide **when to stop**, never to **choose between** candidates:
picking the candidate that looks best on held-out would slowly turn it into another valset.

## 4. What `max_metric_calls` costs

One **metric call** = one document evaluated with one candidate = one extractor call (unless
the disk cache already has it).

Per run: `|valset|` calls once (seed score). Then, per iteration:

| Step | Metric calls |
|---|---|
| parent on minibatch | m (`reflection_minibatch_size`) |
| child on minibatch | m (skipped if the parent minibatch was perfect) |
| child on valset | \|valset\|, only if accepted |

So `iterations ≈ (max_metric_calls − |valset|) / (2m + p·|valset|)` with `p` the acceptance
rate. GEPA checks the budget **between** iterations, so a run may end a few calls over.

USD per call, measured on this project (2026-10-17, Parasail fp8 for Qwen, reasoning off;
Haiku reasoning `medium`):

| Call | Tokens in / out | USD |
|---|---|---|
| extractor (1 page) | ~4.9k / ~175 | ~$0.0016 (~$0.0006 when the provider's prompt cache hits) |
| diagnostic (per failing field, cached across iterations) | ~4.7k / ~230 | ~$0.0006 |
| reflection (per iteration) | ~1–8k / ~1.4k (incl. ~300 thinking) | ~$0.001–0.002 |

Example: `max_metric_calls: 300`, valset 20, m = 3, ~30% acceptance → ~23 iterations →
300 × $0.0016 ≈ **$0.48** for extraction, + ≤ $0.15 diagnostics, + ~$0.05 reflection, +
held-out scoring (seed, best, top-3 × held-out size) → **≈ $0.70–0.90 per run**. Two
guards: `budget.max_usd` (whole run, with `budget.heldout_reserve_usd` kept for held-out
scoring) and `gepa.max_reflection_cost_usd`.

## 5. Reading a run and deciding when to stop

`vex optimize` writes `runs/<timestamp>-optimize-<hash>/`:

| File | What to look at |
|---|---|
| `iterations.jsonl` | `accepted` / `rejected` events with minibatch scores; `end` lines with `total_metric_calls` |
| `proposals.jsonl` | every text the reflection model proposed (even rejected ones) |
| `candidates.jsonl` | each candidate scored on the valset: `average_score`, `parent_ids`, full text |
| `candidates/<i>.json`, `pareto_front.json` | the final pool; which candidates are best on which valset docs |
| `heldout.json` | seed vs best vs top Pareto candidates on `heldout_blind` / `heldout_standin` |
| `cost.json` | USD and tokens per role, cached calls |
| `gepa/run_log.txt`, `gepa/gepa_state.bin` | GEPA's own log; the state lets `vex optimize --resume <dir>` continue |

Patterns, all seen in the notebook:

- **Many `end` lines with `proposal_accepted: false` and no `rejected` event**: GEPA is
  skipping perfect minibatches. The trainset is too easy: annotate more, harder documents
  (`vex select`, P6) rather than raising `max_metric_calls`.
- **Accepted children that all score like the seed on the valset**: the valset is too small
  or too easy to discriminate, and "best" stays the seed (ties keep the first candidate).
- **Many rejections in a row**: the reflection records may be too thin (few diagnoses, many
  dropped). Check `proposals.jsonl` and the suspect-annotation queue.
- **`suspect_annotations.jsonl` growing**: fix the labels first (a new `reviewed` record
  supersedes the old one); optimizing on wrong gold teaches wrong rules.

**When to stop:** when the held-out score of successive rounds stops improving by more than
`tracking.plateau_delta` for `tracking.plateau_rounds` rounds (`vex report`, P6). Within a run,
stop early by creating a `gepa.stop` file in the run's `gepa/` directory.

## 6. Concept → code

| GEPA concept | Where it lives here |
|---|---|
| candidate, components | `vex.prompt.seed_candidate`, `render`, `save_candidate` / `load_candidate` |
| `GEPAAdapter` | `vex.gepa_adapter.VexAdapter` |
| `DataInst` | `vex.gepa_adapter.DocInstance` (doc_id, path, gold values) |
| `evaluate(batch, candidate, capture_traces)` | `VexAdapter.evaluate` → `vex.extract.extract_document` + `vex.metric.compare.score_document` |
| trajectory | dict built in `VexAdapter._trajectory` (raw output, parsed, per-field diffs) |
| score | weighted field accuracy, `vex.metric.compare.score_document` (`metric.weights`) |
| `make_reflective_dataset` | `VexAdapter.make_reflective_dataset` + `vex.diagnostic.Diagnostician` |
| `module_selector` | `vex.gepa_adapter.ErrorDrivenSelector` (or `round_robin` / `all` via YAML) |
| `reflection_lm` | `vex.optimize.ReflectionLM` (Haiku, cost-tracked) |
| proposer / `propose_new_texts` | GEPA's default `InstructionProposalSignature`; extension point `VexAdapter.propose_new_texts = None` |
| `trainset` / `valset` | `vex.annotation.store.effective_split("gepa_train" / "gepa_pareto")` |
| `max_metric_calls`, `max_reflection_cost` | `gepa.max_metric_calls`, `gepa.max_reflection_cost_usd` in YAML |
| stop callbacks | `vex.optimize.BudgetStopper` (`budget.max_usd`), GEPA's `gepa.stop` file |
| callbacks | `vex.optimize.RunRecorder` (writes the `*.jsonl` logs) |
| merge | `gepa.use_merge`, `gepa.max_merge_invocations` |
| `gepa.optimize(...)` call | `vex.optimize.run_optimize` |
| held-out evaluation (not GEPA) | `vex.optimize.score_heldout` |

## 7. Extension points (not implemented)

- **Images in reflection.** GEPA 0.1.4's default proposer already accepts `gepa.Image`
  objects inside reflective records and sends them to the reflection model. Setting
  `gepa.reflection_images: true` adds the first page to each field record. It's off by
  default: the diagnostic model already looks at the image and quotes the zone, and images
  multiply reflection cost.
- **Custom multimodal proposer.** Assign a callable to `VexAdapter.propose_new_texts`
  (`(candidate, reflective_dataset, components) -> dict[str, str]`). GEPA then calls it
  *instead of* its default proposer, e.g. to rewrite several coupled components at once or
  to use a vision model end to end. Out of scope for this project.

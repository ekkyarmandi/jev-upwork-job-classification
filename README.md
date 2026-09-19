# Jev Upwork Job Classification

An Upwork job triage tool for one specific freelancer. It reads every open job in
`data/upwork-jobs.csv`, judges how well that job fits the freelancer's stack, lanes,
rate floor, and evidence, and returns one decision per job: **apply**, **review**,
or **skip**.

The point is not to "let an AI decide". It is the opposite split of
responsibility:

- **Jev owns the semantic read.** It reads one job description and returns typed
  answers: probabilities on yes/no questions, one choice from a fixed list, and a
  graded score on an ordered scale. No generated prose to parse.
- **Code owns every number.** Weights, gate thresholds, fit bands, competition
  penalties, and the decision rule all live in `main.py` as plain Python. You can
  read the arithmetic, argue with it, and change it without touching a prompt.

Built with [TypeSafe AI](https://typesafe.ai) and its flagship System One model,
**Jev**, through the [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python).
Jobs come from [Upwork](https://www.upwork.com).

## What it does

For each job, one Jev request is made. All questions are batched into that single
call, each job classified independently, requests run in parallel.

The pipeline is:

1. **Build the state.** The job row (dotted CSV headers rebuilt into a nested
   object), the client block, and the freelancer baseline are merged into one JSON
   state. Nulls and empty values are pruned; `false` and `0` are kept because they
   are real facts. Descriptions are capped (`--max-chars`, default 12000) because
   accuracy degrades as unrelated detail accumulates.
2. **Ask 13 typed questions.** 4 gates, 7 scores, and 2 choices (lane and work
   type), all defined in `build_questions()`.
3. **Compose the verdict.** `compose()` turns raw answers into a decision using
   code-only logic.

### Project classification

The 13 questions and how their answers become a decision:

**Gates (Noul, yes/no probability)**

| Question | Fires when | Effect |
| --- | --- | --- |
| `is_technical_engagement` | below 0.70 | skip: `no_technical_deliverable` |
| `excludes_solo_freelancer` | above 0.80 | skip: `needs_team_or_onsite` |
| `requires_unpaid_work_sample` | above 0.75 | skip: `unpaid_sample_required` |
| `states_application_instruction` | above 0.50 | informational flag only, routes the post into the proposal writer |

**Score questions (weighted, sum to 1.0)**

| Score | Weight | What it measures |
| --- | --- | --- |
| `core_stack_fit` | 0.30 | how much of the required tech the freelancer has shipped |
| `senior_leverage` | 0.15 | how much architectural judgment the outcome depends on |
| `portfolio_leverage` | 0.13 | how much this strengthens `verified_evidence` |
| `scope_specificity` | 0.12 | how much the post actually tells you what to build |
| `delivery_scale` | 0.10 | size of the build |
| `ai_lane_depth` | 0.10 | how central AI/LLM/automation is to the deliverable |
| `client_decisiveness` | 0.10 | how decided the client is |

Each score is normalized by its level count (`score / (levels - 1)`), multiplied
by its weight, and summed into `fit`.

**Choice questions**

- `primary_lane`: `custom_software_web_app`, `mvp_development`, `api_development`,
  `ai_system_integration`, `python_development`, `technical_review_and_scoping`,
  `outside_my_services`, `unclear_or_mixed`.
- `work_type`: what the client actually wants done (build, integrate, fix, migrate,
  optimize, add feature, scrape, mentor, maintain, unclear).

**Decision rule**

1. **Skip** if any hard gate fires, or if the lane is `outside_my_services`
   with confidence above 0.60.
2. **Contradiction guard.** If only `is_technical_engagement` pushed toward a
   skip, but the lane answer confidently (above 0.70) places the post in a real
   lane, that gate is removed and replaced by a review flag. One low Noul must not
   silently discard a post the lane read disagrees with.
3. `fit >= 0.62` (FIT_APPLY) with no review flags gives **apply**.
4. `fit >= 0.45` (FIT_REVIEW), or any review flag, gives **review**.
5. Everything else is **skip**.

**Soft signals** never auto-skip. They force a human read: `lane_unclear`,
`lane_low_confidence`, `underpriced_for_scope`, `below_rate_floor`,
`payment_unverified`, and the contradiction guard itself.

**Competition** is a compensating preference, not a blocker. Crowded posts
(`50_plus` -0.08, `20_49` -0.03) discount the fit rather than forcing a review, so
`apply` stays reachable on a platform where most posts are crowded.

Every verdict carries a deterministic `reason` string assembled from code, never
generated text: which gate fired, which checks fired, and the two strongest and one
weakest score drivers.

## Clone and set up

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11+ | `uv` can install and pin it for you |
| [uv](https://docs.astral.sh/uv/) | recommended; it reads `uv.lock` for reproducible installs |
| A TypeSafe API key | get one from [typesafe.ai](https://typesafe.ai) |

### 1. Clone

```bash
git clone https://github.com/ekkyarmandi/jev-upwork-job-classification.git
cd jev-upwork-job-classification
```

Over SSH:

```bash
git clone git@github.com:ekkyarmandi/jev-upwork-job-classification.git
cd jev-upwork-job-classification
```

The repo is self-contained. The job data in `data/` is committed, so no upstream
generation step is needed before your first run.

### 2. Install dependencies

```bash
uv sync
```

This creates `.venv/` and installs `typesafe-sdk` and `python-decouple` from
`uv.lock`.

Prefer plain pip? Just install the two dependencies, which is all this project
needs, since `main.py` is a standalone script rather than an installable package:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install typesafe-sdk python-decouple
```

Do not reach for `pip install -e .`. There is no `[build-system]` in
`pyproject.toml`, so an editable install only resolves under a modern pip plus
Python 3.11 or newer, and buys you nothing for a single script.

### 3. Configure your key

```bash
cp .env.example .env
```

Then open `.env` and set:

```
TYPESAFE_API_KEY=your-key-here
JEV_MODEL=jev-latest
```

`.env` is gitignored, so your key is never committed. Billing is **input only** at
`$42 per billion input tokens`; output tokens are free, and Jev's answers are short.

Note: `.env.example` currently ships the model key misspelled as `JEV_MODEl`
(lowercase `l`). It is harmless because `main.py` falls back to the default
`jev-latest`, but `JEV_MODEL=...` is the key that actually takes effect.

### 4. Verify the install

This step needs no API key and makes no network calls, so it is the cheapest way to
confirm the environment is correct:

```bash
uv run python main.py --dry-run --limit 3
```

You should see the merged state for each job printed as JSON, then a list of the 13
questions that would be asked. If you see a `ModuleNotFoundError` here, the
environment is not active: use `uv run` or activate `.venv` first.

Then make a real call:

```bash
uv run python main.py --limit 3
```

### On `uv run`

Every command in this README is written as `uv run python main.py ...`. That
resolves the project environment for you and works from anywhere in the repo. If you
prefer `python main.py` on its own, activate the environment first with
`source .venv/bin/activate`, otherwise Python will not find `typesafe_sdk` or
`python-decouple` and the run will fail on import.

## Data

`data/` is generated, not hand-written. It is produced by
`ekky.dev/scripts/normalize_jev_states.py` (a sibling repo) and committed here so
the classifier runs standalone.

| File | Contents |
| --- | --- |
| `upwork-jobs.csv` | 331 open jobs, one row each, 64 columns, dotted headers are paths into the request state |
| `freelancer.json` | the fixed capability baseline, merged into every state |
| `history.completed.json` | 101 completed contracts plus realized-rate aggregates |
| `manifest.json` | band definitions, column types, coverage stats, and what is computed in code vs asked of Jev |

Sources in the CSV: `healthcare` (12 curated labeled jobs), `fastapi` (316),
`md` (3). Fourteen jobs carry provisional seed labels for accuracy checks.

## Usage

```bash
uv run python main.py                          # classify the 12 curated jobs
uv run python main.py --all --limit 60         # first 60 jobs across every source
uv run python main.py --case-id md:job_1 --verbose
uv run python main.py --labeled                # curated jobs plus accuracy vs seed labels
uv run python main.py --dry-run --limit 3      # build states, print them, call nothing
uv run python main.py --all --json out.json    # write full verdicts and answers
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--source healthcare\|fastapi\|md\|all` | pick a job source (default `healthcare`) |
| `--all` | shortcut for `--source all` |
| `--limit N` | classify at most N jobs |
| `--case-id ID` | classify one case id (repeatable) |
| `--concurrency N` | parallel requests (default 4) |
| `--model NAME` | Jev model (default `jev-latest`) |
| `--max-chars N` | description cap per job (default 12000) |
| `--t1` / `--t2` | fit thresholds for apply / review (default 0.62 / 0.45) |
| `--verbose` | per-score breakdown and weights |
| `--reasons` | print the reason line per job |
| `--json PATH` | write verdicts, answers, weights, and thresholds |

Start with `--dry-run`, then `--labeled`, then widen with `--limit`.

## Changing the input

Most tuning is a one-file edit in `main.py`. Because Jev reads the state and code
owns the numbers, changing a policy value re-ranks everything on the next run
without changing a single question.

**Policy knobs, all near the top of `main.py`:**

- `WEIGHTS` - rebalance the fit. Keep the sum at 1.0.
- `GATE_THRESHOLDS` and `GATE_DIRECTION` - how strict each gate is and which side
  fires it (`high_is_bad` vs low-is-bad).
- `FIT_APPLY` / `FIT_REVIEW`, or pass `--t1` / `--t2` per run.
- `COMPETITION_PENALTY`, `HARD_BANDS_BELOW_PACKAGE_FLOOR`,
  `LANE_CONFIDENCE_FLOOR`, `LANE_OUTSIDE_SKIP_CONFIDENCE`,
  `CONTRADICTION_LANE_CONFIDENCE`, `MAX_DESCRIPTION_CHARS`.

**Questions** live in `build_questions()`. Adding a Score means adding its level
list there and a weight in `WEIGHTS`; `LEVEL_COUNTS` is derived automatically, so
normalization stays in sync. Adding a gate means adding it to `GATE_THRESHOLDS`
and `GATE_DIRECTION`, and referencing it in `compose()`.

### Freelancer detail

This is the highest-leverage input, because it is the only part of the state that
describes *you* and every score is judged relative to it. It is a plain JSON file:
`data/freelancer.json`. Edit it and re-run; nothing else has to change.

The whole file is merged into every request state, so Jev can read all of it. Only
one field is also read by code:

| Field | Code | Jev reads it for |
| --- | --- | --- |
| `rate_floor_usd_per_hour` | yes, drives the `below_rate_floor` review flag against `job.budget.hourly_midpoint_usd` | the same floor, plus rate-fit context |
| `core_stack` | no | `core_stack_fit` levels; the criteria name this field directly |
| `secondary_stack` / `weak_areas` | no | `core_stack_fit`; these name the "Same language, unproven tooling" and "Outside the stack" levels |
| `verified_evidence` | no | `portfolio_leverage`, the "Nothing gained" vs "Flagship case study" split |
| `positioning` | no | lane, scope, and `senior_leverage` reads |
| `availability` | no | `work_type` and retainer reads |
| `target_rate_usd_per_hour` | no | rate-fit context. Not yet a gate or a fit discount |
| `package_floor_usd` | no | fixed-price scope context. The code band check uses `HARD_BANDS_BELOW_PACKAGE_FLOOR` instead |
| `timezone` | no | currently `null`; available for future locality checks |
| `evidence_receipts` | no | concrete proof for proposal writing, not scored |
| `lane_evidence_counts` | no | which lanes are backed by repetition in past work |

Two things to watch:

- `core_stack`, `secondary_stack`, and `weak_areas` are the strings the Score
  criteria literally reference by name. If you rename a technology, update the
  corresponding level `signals` in `build_questions()` so the two stay
  consistent.
- Editing a context-only field changes Jev's read without changing any code, so
  re-run `--labeled` to see whether accuracy moved. `provisional` lists the fields
  that were guessed rather than measured, and `notes` explains that
  `lane_evidence_counts` and `weak_area_mentions_in_past_work` are keyword counts
  over contract titles: they show repetition, not outcomes. Correct `provisional`
  first, then trust the accuracy numbers.

To change the shape of the state itself - new job fields, a different client
block, different band definitions - edit `build_state()` / `prune()` in `main.py`
and the matching CSV column-type sets (`CSV_LIST_COLUMNS`, `CSV_BOOL_COLUMNS`,
`CSV_NUMERIC_COLUMNS`). Regenerating the CSV itself happens upstream in
`ekky.dev/scripts/normalize_jev_states.py`.

## Cost and limits

- One request per job, 13 questions batched in. Input-only billing.
- `--dry-run` builds and prints states without calling the API, so you can inspect
  the exact payload for free.
- Exit codes: `0` clean, `1` API errors occurred, `2` bad input.

## References

- TypeSafe AI - [typesafe.ai](https://typesafe.ai)
- TypeSafe docs - [docs.typesafe.ai](https://docs.typesafe.ai)
- Python SDK - [docs.typesafe.ai/sdk/python](https://docs.typesafe.ai/sdk/python)
- System One concepts - [docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one)
- Upwork - [upwork.com](https://www.upwork.com)

#!/usr/bin/env python3
"""Jev-based Upwork job classifier.

One Jev request per job, all Round-1 questions batched into that call. Code owns
gates, weights, thresholds, and every number. Jev owns the semantic read.

Data comes from ./data, produced by ekky.dev/scripts/normalize_jev_states.py:
  upwork-jobs.csv        all 331 open jobs, one row each, labels folded in
  freelancer.json        fixed capability baseline, merged into every state
  manifest.json          band definitions, column types, coverage stats
  history.completed.json completed contracts and realized-rate aggregates

Examples:
  python main.py                      # classify the 12 curated jobs
  python main.py --all --limit 60     # first 60 jobs across every source
  python main.py --case-id md:job_1 --verbose
  python main.py --labeled            # 12 curated jobs plus accuracy vs seeds
  python main.py --dry-run --limit 3  # build states, print them, call nothing
  python main.py --all --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decouple import config

from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeClient,
    TypeSafeError,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Policy. Every number here is a decision, not a fact. Change them in code and
# re-rank; the questions and the state do not need to change.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = config("JEV_MODEL", default="jev-latest")

# Input-only billing. Jev charges per input token and output tokens are free.
INPUT_USD_PER_BTOK = 42.0

# Score weights. Sum to 1.0 over the seven Score questions.
WEIGHTS: dict[str, float] = {
    "core_stack_fit": 0.30,
    "senior_leverage": 0.15,
    "scope_specificity": 0.12,
    "delivery_scale": 0.10,
    "ai_lane_depth": 0.10,
    "portfolio_leverage": 0.13,
    "client_decisiveness": 0.10,
}

# Noul thresholds. `direction` says which side of the threshold fires the gate.
# `high_is_bad` means the gate fires when the value is ABOVE the threshold.
GATE_THRESHOLDS: dict[str, float] = {
    "is_technical_engagement": 0.70,
    "excludes_solo_freelancer": 0.80,
    "requires_unpaid_work_sample": 0.75,
}
GATE_DIRECTION: dict[str, bool] = {
    # False: fires when the value is BELOW the threshold.
    "is_technical_engagement": False,
    "excludes_solo_freelancer": True,
    "requires_unpaid_work_sample": True,
}

def gate_fired(name: str, value: float) -> bool:
    """One place that decides whether a gate condition is met."""
    if GATE_DIRECTION[name]:
        return value > GATE_THRESHOLDS[name]
    return value < GATE_THRESHOLDS[name]


# Rewriting the state is impossible, so a lone low gate score that contradicts a
# confident lane answer is treated as a contradiction, not a verdict.
CONTRADICTION_LANE_CONFIDENCE = 0.70

# Fit bands for the decision.
FIT_APPLY = 0.62
FIT_REVIEW = 0.45

# A lane answer split across options is a signal to read the post yourself.
LANE_CONFIDENCE_FLOOR = 0.50
# Only skip on an out-of-scope lane when the model is reasonably sure.
LANE_OUTSIDE_SKIP_CONFIDENCE = 0.60

# Competition is a compensating preference, not a blocker: a crowded post is
# worse, but a strong one is still worth a Connect. It discounts the fit rather
# than forcing a review, so `apply` stays reachable on this platform.
COMPETITION_PENALTY: dict[str, float] = {"50_plus": 0.08, "20_49": 0.03}

# 32k tokens of state plus the longest question is the hard limit. Accuracy also
# falls as unrelated detail accumulates, so this cap is deliberately well under.
MAX_DESCRIPTION_CHARS = 12_000

HARD_BANDS_BELOW_PACKAGE_FLOOR = {"under_500", "500_1500"}

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def build_questions() -> tuple[dict[str, Any], dict[str, int]]:
    """Round-1 questions, all evaluated in parallel in one request.

    Returns the question mapping and the level count per Score, so normalization
    stays in sync with the criteria.
    """
    score_levels: dict[str, list[Any]] = {
        "core_stack_fit": [
            {
                "summary": "Outside the freelancer's stack entirely",
                "signals": [
                    "Ruby on Rails, PHP/Laravel, .NET/C#, Java/Spring, or Go as the primary language",
                    "native Swift or Kotlin",
                    "Salesforce, SAP, or Dynamics platform work",
                    "smart contracts",
                ],
            },
            {
                "summary": "Same language, unproven tooling",
                "signals": [
                    "a frontend framework other than React or Next.js",
                    "WordPress theme or plugin internals",
                    "owning Kubernetes or Terraform",
                    "a specialized machine learning or numerical stack",
                ],
            },
            {
                "summary": "Mixed",
                "signals": [
                    "at least one core requirement is in `freelancer.core_stack` and at least one is not"
                ],
            },
            {
                "summary": "Mostly core stack",
                "signals": [
                    "all main requirements are in `freelancer.core_stack`, one or two tools come from `freelancer.secondary_stack` or `freelancer.weak_areas`"
                ],
            },
            {
                "summary": "Entirely core stack",
                "signals": [
                    "every tool and language named appears in `freelancer.core_stack`"
                ],
            },
        ],
        "ai_lane_depth": [
            {
                "summary": "No AI or automation content",
                "signals": ["conventional application code only"],
            },
            {
                "summary": "AI as vocabulary only",
                "signals": [
                    "'AI-friendly' or 'interested in AI' with no AI deliverable",
                    "a future phase that may add AI later",
                ],
            },
            {
                "summary": "AI as one feature",
                "signals": [
                    "a single LLM call or classification step",
                    "one chatbot widget",
                    "one n8n workflow beside a larger app",
                ],
            },
            {
                "summary": "AI or automation is the product",
                "signals": [
                    "retrieval over the client's own data",
                    "a multi-step agent or pipeline",
                    "an MCP server",
                    "evals, or prompt and model behavior as the deliverable",
                ],
            },
        ],
        "senior_leverage": [
            {
                "summary": "Mechanical and fully specified",
                "signals": [
                    "a defined edit or narrow bugfix",
                    "a routine CRUD endpoint",
                    "a script whose behavior is written out in the post",
                ],
            },
            {
                "summary": "Competent execution, no architecture",
                "signals": [
                    "a standard feature added inside an existing well-structured codebase"
                ],
            },
            {
                "summary": "Bounded design decisions",
                "signals": [
                    "schema design",
                    "API contract design",
                    "choosing an integration approach",
                    "performance work against a stated target",
                ],
            },
            {
                "summary": "Architecture decides success",
                "signals": [
                    "greenfield system",
                    "migration off a fragile base",
                    "an AI system whose data, retrieval, and eval design determines whether it works at all",
                ],
            },
        ],
        "scope_specificity": [
            {
                "summary": "Nothing concrete",
                "signals": [
                    "no deliverable, no users, no stack",
                    "asks for ideas or a plan instead of a build",
                ],
            },
            {
                "summary": "Goal only",
                "signals": [
                    "a stated outcome with no deliverables, stack, or acceptance criteria"
                ],
            },
            {
                "summary": "Deliverables and stack clear",
                "signals": [
                    "what to build and with what is stated",
                    "milestones, acceptance criteria, or constraints are loose or absent",
                ],
            },
            {
                "summary": "Quotable as written",
                "signals": [
                    "deliverables, stack, constraints, and acceptance criteria are all explicit",
                    "a fixed price could be quoted with no discovery call",
                ],
            },
        ],
        "delivery_scale": [
            {
                "summary": "One small change",
                "signals": [
                    "a single bugfix, endpoint, page, or script with no surrounding system to consider"
                ],
            },
            {
                "summary": "One bounded feature",
                "signals": [
                    "one subsystem, one integration, or one screen set on an existing product"
                ],
            },
            {
                "summary": "A multi-feature build",
                "signals": [
                    "several screens plus backend work plus authentication inside one application"
                ],
            },
            {
                "summary": "A multi-module product",
                "signals": [
                    "authentication, payments or billing, several user roles, and admin tooling in one engagement"
                ],
            },
            {
                "summary": "A platform program",
                "signals": [
                    "several applications or channels delivered together, such as web plus mobile plus integrations plus a data or AI layer"
                ],
            },
        ],
        "portfolio_leverage": [
            {
                "summary": "Nothing gained",
                "signals": [
                    "routine work in territory already better covered by current evidence"
                ],
            },
            {
                "summary": "Another example, same territory",
                "signals": [
                    "work already covered by an existing item in `freelancer.verified_evidence`"
                ],
            },
            {
                "summary": "Strong new case study",
                "signals": [
                    "a named outcome in AI systems, API and backend work, or MVP delivery, in a domain the freelancer has not covered"
                ],
            },
            {
                "summary": "Flagship case study",
                "signals": [
                    "a production system whose results no current evidence can demonstrate, such as a retrieval system serving a named industry"
                ],
            },
        ],
        "client_decisiveness": [
            {
                "summary": "Exploring",
                "signals": [
                    "asks for options, ideas, or feasibility input before any scope exists"
                ],
            },
            {
                "summary": "Has a goal, no decisions",
                "signals": ["no chosen stack, no budget anchor, no timeline"],
            },
            {
                "summary": "Decided",
                "signals": [
                    "names the stack, the deliverable, and either a timeline or a budget"
                ],
            },
            {
                "summary": "Committed",
                "signals": [
                    "names the stack, the deliverable, a timeline or budget, and either an existing codebase or a prior hire"
                ],
            },
        ],
    }

    questions: dict[str, Any] = {
        # ---- gates ------------------------------------------------------
        "is_technical_engagement": Noul(
            instructions={
                "question": "Does `job.description` offer technical work the freelancer can be paid to deliver?",
                "focus": (
                    "Judge the deliverable, not the industry or product it serves. "
                    "Implementation is one kind of technical deliverable; a paid review of a "
                    "system the freelancer understands is another."
                ),
            },
            criteria={
                "true": {
                    "what": (
                        "Implementation is in scope, OR the client will pay for a written or verbal "
                        "technical artifact that requires reading and judging a real system: an "
                        "architecture or design review, a code audit, a technical scoping document, "
                        "a feasibility assessment, or a performance diagnosis"
                    ),
                    "examples": [
                        "Build a FastAPI backend for our mobile app",
                        "Fix our Stripe webhook handler",
                        "Review our multi-tenant architecture and write a prioritized action plan",
                        "Audit our codebase and tell us what to fix before launch",
                    ],
                },
                "false": {
                    "what": (
                        "No technical deliverable. The work is design, branding, writing, marketing, "
                        "admin, data entry, or non-technical advice. This also covers teaching "
                        "beginners fundamentals, and formal security testing or compliance "
                        "certification that requires a licensed or certified practitioner"
                    ),
                    "examples": [
                        "Design a logo and brand kit",
                        "Write 10 SEO blog posts",
                        "Act as our fractional CTO in meetings only",
                        "Teach me terminal, Git, and GitHub from zero",
                        "Perform a penetration test and sign a security attestation",
                    ],
                },
            },
        ),
        "excludes_solo_freelancer": Noul(
            instructions=(
                "Does `job.description` require an agency, several developers working at "
                "the same time, an on-site presence, or a legally registered company "
                "rather than one person?"
            ),
            criteria={
                "true": {
                    "what": "A team, agency, or simultaneous multi-role staffing is required",
                    "examples": [
                        "We need a team of 3 developers",
                        "You will manage 4 engineers",
                        "Must be available on-site in Austin",
                    ],
                },
                "false": {
                    "what": "One competent developer can deliver the whole engagement",
                    "examples": [
                        "Looking for a developer to own this build",
                        "We have a designer, we need the build",
                    ],
                },
            },
        ),
        "requires_unpaid_work_sample": Noul(
            instructions=(
                "Does `job.description` make a free sample, an unpaid test project, or "
                "an unpaid trial period a condition of being hired?"
            ),
            criteria={
                "true": {
                    "what": "Completing work for no pay is required before or during hiring",
                    "examples": [
                        "Build a small demo app for us to review before we decide",
                        "Two-week unpaid trial",
                    ],
                },
                "false": {
                    "what": "Hiring relies on a portfolio, a paid test, or an interview",
                    "examples": [
                        "Send your GitHub",
                        "We pay a small fixed fee for a trial task",
                    ],
                },
            },
        ),
        "states_application_instruction": Noul(
            instructions={
                "question": "Does `job.description` contain an instruction addressed to the person applying?",
                "focus": (
                    "Look for a word to include, a question to answer in the proposal, or a "
                    "required proposal format. Ignore ordinary requirements about skills or "
                    "experience."
                ),
            },
        ),
    }

    score_instructions = {
        "core_stack_fit": "How much of the technology required by `job.description` is technology the freelancer has shipped?",
        "ai_lane_depth": "How central are AI, LLM, or automation systems to the deliverable in `job.description`?",
        "senior_leverage": "How much does the outcome of `job.description` depend on architectural judgment made before the build starts?",
        "scope_specificity": "How much does `job.description` tell a senior engineer about what to build?",
        "delivery_scale": "How large is the build described in `job.description`?",
        "portfolio_leverage": "How much would shipping this engagement strengthen `freelancer.verified_evidence` for the lanes the freelancer is trying to sell?",
        "client_decisiveness": "How decided is the client, judging only from `job.description`?",
    }
    for name, levels in score_levels.items():
        questions[name] = Score(instructions=score_instructions[name], criteria=levels)

    questions["primary_lane"] = Choice(
        instructions={
            "question": "Which of the freelancer's service lanes does the main deliverable in `job.description` belong to?",
            "focus": (
                "Choose the lane that owns the largest part of the work and the outcome "
                "the client is paying for, not every technology mentioned."
            ),
        },
        criteria={
            "custom_software_web_app": {
                "what": "Business web app, internal tool, client portal, ops dashboard, admin panel, or SaaS platform built or extended for a specific workflow",
                "not_for": "A first-time product validation build, a standalone API with no UI, or an AI system as the core deliverable",
                "examples": [
                    "Rebuild our ops dashboard",
                    "Replace our spreadsheet workflow with an internal tool",
                ],
            },
            "mvp_development": {
                "what": "A first shippable version of a new product on a live URL, from scoping and design through build and deployment",
                "not_for": "Extending an established product, or work on a system the client already runs in production",
                "examples": [
                    "I have an idea and need a working product to show investors",
                    "Ship v1 of our marketplace",
                ],
            },
            "api_development": {
                "what": "REST or GraphQL APIs, partner-facing endpoints, webhooks, third-party integrations, background workers, and backend services",
                "not_for": "A full product with its own UI as the main deliverable, or AI or model behavior as the core problem",
                "examples": [
                    "Build a REST API for our mobile app",
                    "Integrate Stripe and our CRM",
                ],
            },
            "ai_system_integration": {
                "what": "LLM features in a product: retrieval over the client's data, agents, pipelines, MCP servers, evals, model output surfaced in a UI",
                "not_for": "A conventional app where the model is incidental, or prompt writing with no system around it",
                "examples": [
                    "Add a RAG assistant over our documentation",
                    "Build an agent that qualifies inbound leads",
                ],
            },
            "python_development": {
                "what": "Python work where the language or the automation is the point: scraping, data pipelines, ETL, scheduled jobs, scripts, workflow automation",
                "not_for": "Work where Python is an implementation detail of a larger product build already covered by another lane",
                "examples": [
                    "Scrape 40 supplier sites nightly",
                    "Automate our reporting with Python and n8n",
                ],
            },
            "outside_my_services": {
                "what": "The main deliverable falls outside all five lanes, or the work is design, writing, marketing, admin, data entry, or a non-code service",
                "not_for": "Any engagement where the bulk of the paid work is one of the five lanes above",
            },
            "technical_review_and_scoping": {
                "what": "Paid technical judgment with no implementation: architecture or design review, code audit, technical scoping document, feasibility assessment, or performance diagnosis on a system the freelancer understands",
                "not_for": "Building or changing the system, ongoing advisory with no artifact, penetration testing, or compliance attestation",
                "examples": [
                    "Review our multi-tenant architecture and write a prioritized action plan",
                    "Audit this codebase before we hire a team",
                    "Write a written scope and fixed-price proposal for our platform",
                ],
            },
            "unclear_or_mixed": {
                "what": "The post does not say enough to place the work, or two lanes are genuinely equal and the largest part cannot be identified"
            },
        },
    )

    questions["work_type"] = Choice(
        instructions="What does the client actually want done, based on `job.description`?",
        criteria={
            "build_new_from_scratch": "A new application, platform, or system with nothing existing to extend",
            "build_ai_or_automation_solution": "An AI, chatbot, RAG, or automation system as the thing being delivered",
            "integrate_third_party": "Connect systems, APIs, or services that already exist",
            "fix_or_troubleshoot": "Diagnose and repair something that is broken",
            "migrate_or_upgrade": "Move or modernize an existing system",
            "optimize_performance": "Make an existing system faster, cheaper, or more reliable",
            "add_feature": "Extend a product that is already in use",
            "scrape_or_process_data": "Extract, transform, or process data",
            "mentorship_or_training": "Teach, review, or advise rather than build",
            "maintain_or_support": "Ongoing upkeep, on-call, or team augmentation",
            "unclear_or_mixed": "The post mixes several of these or does not resolve to one",
        },
    )

    level_counts = {name: len(levels) for name, levels in score_levels.items()}
    return questions, level_counts


QUESTIONS, LEVEL_COUNTS = build_questions()

# ---------------------------------------------------------------------------
# Environment and data loading
# ---------------------------------------------------------------------------


def load_json(name: str) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "run: python ../ekky.dev/scripts/normalize_jev_states.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# CSV cell types. Header names are dotted paths into the nested state, so the file
# is self-describing. Anything not listed here stays text, which keeps free text
# like a numeric-looking title from being coerced.
CSV_LIST_COLUMNS = {
    "job.client_stated_deliverables",
    "job.required_qualifications",
    "job.skills.mandatory",
    "job.skills.nice_to_have",
    "job.skills.general",
    "job.location_restrictions",
    "job.attachments",
    "client.signals",
}
CSV_BOOL_COLUMNS = {
    "job.enterprise_job",
    "job.premium",
    "client.payment_verified",
    "seed.is_technical_engagement",
    "seed.hard_case",
}
CSV_NUMERIC_COLUMNS = {
    "job.description_chars",
    "job.budget.hourly_min_usd",
    "job.budget.hourly_max_usd",
    "job.budget.hourly_midpoint_usd",
    "job.budget.fixed_usd",
    "job.competition.interviews",
    "job.competition.invites_sent",
    "job.posted_age_days",
    "job.connects_required",
    "job.bid_range.high_usd",
    "job.bid_range.average_usd",
    "job.bid_range.low_usd",
    "job.persons_to_hire",
    "client.member_days",
    "client.rating",
    "client.reviews",
    "client.total_spend_usd",
    "client.hires",
    "client.active_hires",
    "client.jobs_posted",
    "client.hire_rate_pct",
    "client.open_jobs",
    "client.avg_hourly_paid_usd",
}


def coerce(column: str, value: str) -> Any:
    """Turn one CSV cell into a typed value, or None when the source was empty."""
    if value == "":
        return None
    if column in CSV_LIST_COLUMNS:
        return json.loads(value)
    if column in CSV_BOOL_COLUMNS:
        return value == "true"
    if column in CSV_NUMERIC_COLUMNS:
        return int(value) if value.lstrip("-").isdigit() else float(value)
    return value


def unflatten(row: dict[str, str]) -> dict:
    """Rebuild the nested entry from dotted headers."""
    entry: dict[str, Any] = {}
    for column, raw in row.items():
        value = coerce(column, raw)
        if value is None:
            continue
        node = entry
        parts = column.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    entry.setdefault("job", {})
    entry.setdefault("client", {})
    return entry


def load_jobs() -> list[dict]:
    path = DATA_DIR / "upwork-jobs.csv"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "run: python ../ekky.dev/scripts/normalize_jev_states.py"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return [unflatten(row) for row in csv.DictReader(handle)]


def prune(value: Any) -> Any:
    """Drop null and empty values so the state carries no filler.

    Accuracy falls as state accumulates unrelated detail, and every empty field
    is noise. False and 0 are kept: they are real facts.
    """
    if isinstance(value, dict):
        return {
            key: pruned
            for key, item in value.items()
            if (pruned := prune(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [item for item in (prune(v) for v in value) if item not in (None, "", [], {})]
    return value


def build_state(entry: dict, freelancer: dict, max_chars: int) -> tuple[dict, bool]:
    job = dict(entry["job"])
    description = job.get("description") or ""
    truncated = len(description) > max_chars
    if truncated:
        cut = description[:max_chars]
        job["description"] = cut[: cut.rfind("\n")] + "\n[description truncated]"
    state = {"job": job, "client": entry["client"], "freelancer": freelancer}
    return prune(state), truncated


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    case_id: str
    title: str
    decision: str
    fit: float
    lane: str
    lane_confidence: float
    work_type: str
    raw_fit: float = 0.0
    penalty: float = 0.0
    application_instruction: float = 0.0
    hard: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    gates: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str | None = None
    answers: dict[str, Any] = field(default_factory=dict)


def compose(entry: dict, answers: dict[str, Any], truncated: bool) -> Verdict:
    """Turn typed answers into one decision. All arithmetic lives here."""
    job = entry["job"]
    client = entry["client"]

    gates = {name: float(answers[name].noul) for name in GATE_THRESHOLDS}

    # Informational, not a gate. It routes the post into the proposal writer.
    application_instruction = float(answers["states_application_instruction"].noul)

    normalized = {
        name: answers[name].score / (LEVEL_COUNTS[name] - 1) for name in WEIGHTS
    }
    fit = round(sum(WEIGHTS[name] * value for name, value in normalized.items()), 4)

    lane = answers["primary_lane"].choice
    lane_confidence = float(answers["primary_lane"].confidence)
    work_type = answers["work_type"].choice

    flags: list[str] = []

    # --- hard gates: any one of these is a skip ---------------------------
    hard: list[str] = []
    if gate_fired("is_technical_engagement", gates["is_technical_engagement"]):
        hard.append("no_technical_deliverable")
    if gate_fired("excludes_solo_freelancer", gates["excludes_solo_freelancer"]):
        hard.append("needs_team_or_onsite")
    if gate_fired("requires_unpaid_work_sample", gates["requires_unpaid_work_sample"]):
        hard.append("unpaid_sample_required")
    if lane == "outside_my_services" and lane_confidence > LANE_OUTSIDE_SKIP_CONFIDENCE:
        hard.append("lane_outside_services")

    # Contradiction guard. A single low Noul must not silently discard a post that
    # the lane answer places confidently in a real lane. Downgrade to a review.
    contradictions: list[str] = []
    if "no_technical_deliverable" in hard and lane not in (
        "outside_my_services",
        "unclear_or_mixed",
    ) and lane_confidence > CONTRADICTION_LANE_CONFIDENCE:
        hard.remove("no_technical_deliverable")
        contradictions.append(
            f"gate_and_lane_disagree(lane={lane} {lane_confidence:.2f})"
        )

    # --- soft signals: force a human read, never auto-skip ----------------
    review: list[str] = list(contradictions)
    if lane == "unclear_or_mixed":
        review.append("lane_unclear")
    if lane_confidence < LANE_CONFIDENCE_FLOOR:
        review.append("lane_low_confidence")

    band = job.get("budget", {}).get("band")
    delivery_scale = normalized["delivery_scale"] * (LEVEL_COUNTS["delivery_scale"] - 1)
    if delivery_scale >= 3 and band in HARD_BANDS_BELOW_PACKAGE_FLOOR:
        review.append(f"underpriced_for_scope({band})")

    hourly_midpoint = job.get("budget", {}).get("hourly_midpoint_usd")
    rate_floor = entry.get("_rate_floor")
    if hourly_midpoint and rate_floor and hourly_midpoint < rate_floor:
        review.append(f"below_rate_floor(${hourly_midpoint:g}<${rate_floor:g})")

    if client.get("payment_verified") is False:
        review.append("payment_unverified")

    penalty = COMPETITION_PENALTY.get(job.get("competition", {}).get("proposals_band", ""), 0.0)
    effective_fit = round(max(fit - penalty, 0.0), 4)

    if application_instruction > 0.5:
        flags.append("post_has_application_instruction")

    if truncated:
        flags.append("description_truncated")

    if hard:
        decision = "skip"
    elif effective_fit >= FIT_APPLY and not review:
        decision = "apply"
    elif effective_fit >= FIT_REVIEW or review:
        decision = "review"
    else:
        decision = "skip"

    # Deterministic explanation, assembled from code. Never generated text.
    parts: list[str] = []
    if hard:
        parts.append("gate: " + ", ".join(hard))
    if review:
        parts.append("check: " + ", ".join(review))
    if penalty:
        parts.append(f"fit {fit:.2f} discounted {penalty:.2f} for competition")
    strongest = sorted(
        ((WEIGHTS[name] * value, name) for name, value in normalized.items()), reverse=True
    )[:2]
    parts.append(
        "drivers: " + ", ".join(f"{name} {value:.2f}" for value, name in strongest)
    )
    weakest = min((normalized[name], name) for name in normalized)
    parts.append(f"weakest: {weakest[1]} {weakest[0]:.2f}")

    return Verdict(
        case_id=entry["case_id"],
        title=job.get("title", ""),
        decision=decision,
        fit=effective_fit,
        raw_fit=fit,
        penalty=penalty,
        lane=lane,
        lane_confidence=lane_confidence,
        work_type=work_type,
        gates=gates,
        application_instruction=application_instruction,
        hard=hard,
        review=review,
        normalized={k: round(v, 3) for k, v in normalized.items()},
        flags=flags,
        reason=" | ".join(parts),
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_one(
    client: TypeSafeClient,
    entry: dict,
    freelancer: dict,
    max_chars: int,
) -> Verdict:
    state, truncated = build_state(entry, freelancer, max_chars)
    try:
        response = client.system_one(state=state, questions=QUESTIONS)
    except TypeSafeError as error:
        return Verdict(
            case_id=entry["case_id"],
            title=entry["job"].get("title", ""),
            decision="error",
            fit=0.0,
            lane="",
            lane_confidence=0.0,
            work_type="",
            reason=f"{type(error).__name__}: {error}",
            error=str(error),
        )

    verdict = compose(entry, dict(response.answers), truncated)
    verdict.input_tokens = response.usage.input_tokens
    verdict.output_tokens = response.usage.output_tokens
    verdict.model = response.model
    verdict.answers = {
        name: (
            {"noul": round(float(answer.noul), 4)}
            if answer.type == "noul"
            else {
                "choice": answer.choice,
                "confidence": round(float(answer.confidence), 4),
                "probabilities": {
                    k: round(float(v), 4) for k, v in answer.probabilities.items()
                },
            }
            if answer.type == "choice"
            else {
                "score": round(float(answer.score), 4),
                "confidence": round(float(answer.confidence), 4),
                "probabilities": {
                    k: round(float(v), 4) for k, v in answer.probabilities.items()
                },
            }
        )
        for name, answer in response.answers.items()
    }
    return verdict


DECISION_ORDER = {"apply": 0, "review": 1, "skip": 2, "error": 3}


def render(verdicts: list[Verdict], verbose: bool, show_reason: bool) -> None:
    ranked = sorted(verdicts, key=lambda v: (DECISION_ORDER[v.decision], -v.fit))
    width = max((len(v.title) for v in ranked), default=20)
    width = min(max(width, 24), 74)

    print()
    print(
        f"{'DECISION':<9} {'FIT':>5}  {'LANE':<22} {'CNF':>4}  {'GATES':<38} TITLE"
    )
    print("-" * (9 + 1 + 5 + 2 + 22 + 1 + 4 + 2 + 38 + 1 + width))
    for verdict in ranked:
        if verdict.decision == "error":
            print(f"{'error':<9} {'-':>5}  {'-':<22} {'-':>4}  {'-':<38} {verdict.title[:width]}")
            if show_reason:
                print(f"{'':<9} {verdict.reason}")
            continue
        # Hard reasons are authoritative: they include lane-derived skips that no
        # single Noul expresses.
        gate_text = ",".join(verdict.hard) or "-"
        print(
            f"{verdict.decision:<9} {verdict.fit:>5.2f}  {verdict.lane:<22} "
            f"{verdict.lane_confidence:>4.2f}  {gate_text[:38]:<38} {verdict.title[:width]}"
        )
        if verbose:
            print(f"{'':<9}   raw fit {verdict.raw_fit:.2f}")
            for name, value in sorted(verdict.normalized.items(), key=lambda kv: -kv[1]):
                print(f"{'':<9}   {name:<22} {value:.2f}  w={WEIGHTS[name]:.2f}")
            if verdict.flags:
                print(f"{'':<9}   flags: {', '.join(verdict.flags)}")
        if show_reason:
            print(f"{'':<9}   {verdict.reason}")


def evaluate(verdicts: list[Verdict], seeds: list[dict]) -> None:
    by_id = {v.case_id: v for v in verdicts}
    seeds_in_run = [s for s in seeds if s["case_id"] in by_id]
    if not seeds_in_run:
        print("\nno seeded labels in this run; add --source healthcare or --all")
        return

    hard_rows, soft_rows = [], []

    for seed in seeds_in_run:
        verdict = by_id[seed["case_id"]]
        predicted = not gate_fired(
            "is_technical_engagement", verdict.gates["is_technical_engagement"]
        )
        code_ok = predicted == seed["seed_is_technical_engagement"]
        lane_ok = verdict.lane == seed["seed_primary_lane"]
        row = (seed, verdict, code_ok, lane_ok)
        (hard_rows if seed["hard_case"] else soft_rows).append(row)

    def report(name: str, subset: list) -> None:
        if not subset:
            return
        code = sum(1 for _, _, ok, _ in subset if ok)
        lane = sum(1 for _, _, _, ok in subset if ok)
        print(
            f"  {name:<12} n={len(subset):<3} is_technical_engagement "
            f"{code}/{len(subset)}  primary_lane {lane}/{len(subset)}"
        )
        for seed, verdict, code_ok, lane_ok in subset:
            if code_ok and lane_ok:
                continue
            marks = []
            if not code_ok:
                marks.append(
                    f"technical: expected {seed['seed_is_technical_engagement']} "
                    f"noul={verdict.gates['is_technical_engagement']:.2f}"
                )
            if not lane_ok:
                marks.append(f"lane: expected {seed['seed_primary_lane']} got {verdict.lane}")
            print(f"      - {seed['title'][:70]}")
            print(f"        {'; '.join(marks)}")

    print("\nseeded labels (provisional, correct them before trusting these numbers)")
    report("all", soft_rows + hard_rows)
    report("clear cases", soft_rows)
    report("hard cases", hard_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Upwork jobs with Jev.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 clean, 1 API errors occurred, 2 bad input\n"
            "cost: Jev bills input only, $42 per billion tokens. Output is free."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["healthcare", "fastapi", "md", "all"],
        default="healthcare",
        help="which normalized job source to classify (default: healthcare, the 12 curated jobs)",
    )
    parser.add_argument("--all", action="store_true", help="shortcut for --source all")
    parser.add_argument("--limit", type=int, default=None, help="classify at most N jobs")
    parser.add_argument("--case-id", action="append", default=None, help="classify one case id")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel requests (default 4)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_DESCRIPTION_CHARS,
        help=f"description cap per job (default {MAX_DESCRIPTION_CHARS})",
    )
    parser.add_argument("--json", metavar="PATH", help="write full verdicts and answers as JSON")
    parser.add_argument("--verbose", action="store_true", help="print per-score breakdown")
    parser.add_argument("--reasons", action="store_true", help="print the reason line per job")
    parser.add_argument("--labeled", action="store_true", help="score against seed_labels.json")
    parser.add_argument("--dry-run", action="store_true", help="build and print states, call nothing")
    parser.add_argument("--t1", type=float, default=FIT_APPLY, help="fit at or above this is apply")
    parser.add_argument("--t2", type=float, default=FIT_REVIEW, help="fit at or above this is review")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global FIT_APPLY, FIT_REVIEW
    FIT_APPLY, FIT_REVIEW = args.t1, args.t2

    freelancer = load_json("freelancer.json")
    entries: list[dict] = load_jobs()

    source = "all" if args.all else args.source
    if args.case_id:
        wanted = set(args.case_id)
        entries = [e for e in entries if e["case_id"] in wanted]
        missing = wanted - {e["case_id"] for e in entries}
        if missing:
            print(f"unknown case ids: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    elif source != "all":
        entries = [e for e in entries if e.get("source", {}).get("kind") == source]

    if args.limit:
        entries = entries[: args.limit]

    if not entries:
        print("no jobs matched", file=sys.stderr)
        return 2

    rate_floor = freelancer.get("rate_floor_usd_per_hour")
    for entry in entries:
        entry["_rate_floor"] = rate_floor

    length_estimate = sum(len(e["job"].get("description") or "") for e in entries)
    print(
        f"{len(entries)} job(s), model={args.model}, concurrency={args.concurrency}, "
        f"~{length_estimate // 4:,} tokens of job text"
    )

    if args.dry_run:
        for entry in entries[:3]:
            state, truncated = build_state(entry, freelancer, args.max_chars)
            print(f"\n--- {entry['case_id']} (truncated={truncated}) ---")
            print(json.dumps(state, indent=2, ensure_ascii=False)[:2500])
        print(f"\n{len(QUESTIONS)} questions would be asked per job:")
        for name, question in QUESTIONS.items():
            kind = getattr(question, "type", "?")
            print(f"  {name:<32} {kind}")
        return 0

    api_key = config("TYPESAFE_API_KEY", default=None)
    if not api_key:
        print(f"no API key. set TYPESAFE_API_KEY in .env", file=sys.stderr)
        return 2

    retry = RetryPolicy(max_retries=4, backoff_initial=1.0, backoff_max=20.0)
    started = time.time()
    verdicts: list[Verdict] = []

    with TypeSafeClient(api_key=api_key, model=args.model, retry=retry) as client:
        if args.concurrency <= 1:
            for index, entry in enumerate(entries, 1):
                verdict = classify_one(client, entry, freelancer, args.max_chars)
                verdicts.append(verdict)
                print(f"  [{index}/{len(entries)}] {verdict.decision:<8} {verdict.title[:60]}")
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(classify_one, client, entry, freelancer, args.max_chars): entry
                    for entry in entries
                }
                for index, future in enumerate(as_completed(futures), 1):
                    verdict = future.result()
                    verdicts.append(verdict)
                    print(f"  [{index}/{len(entries)}] {verdict.decision:<8} {verdict.title[:60]}")

    elapsed = time.time() - started
    render(verdicts, verbose=args.verbose, show_reason=args.reasons)

    if args.labeled:
        seeds = []
        for entry in entries:
            seed = entry.get("seed")
            if not seed or "primary_lane" not in seed:
                continue
            seeds.append(
                {
                    "case_id": entry["case_id"],
                    "title": entry["job"].get("title", ""),
                    "seed_is_technical_engagement": seed.get("is_technical_engagement"),
                    "seed_primary_lane": seed.get("primary_lane"),
                    "hard_case": bool(seed.get("hard_case")),
                    "why": seed.get("why", ""),
                }
            )
        evaluate(verdicts, seeds)

    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.decision] = counts.get(verdict.decision, 0) + 1
    input_tokens = sum(v.input_tokens for v in verdicts)
    output_tokens = sum(v.output_tokens for v in verdicts)
    cost = input_tokens / 1_000_000_000 * INPUT_USD_PER_BTOK

    print()
    print(
        f"{counts.get('apply', 0)} apply, {counts.get('review', 0)} review, "
        f"{counts.get('skip', 0)} skip, {counts.get('error', 0)} error"
    )
    print(
        f"{input_tokens:,} input tokens, {output_tokens:,} output tokens (free), "
        f"${cost:.6f}, {elapsed:.1f}s"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": verdicts[0].model if verdicts else args.model,
                    "fit_apply": FIT_APPLY,
                    "fit_review": FIT_REVIEW,
                    "weights": WEIGHTS,
                    "verdicts": [vars(v) for v in verdicts],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())

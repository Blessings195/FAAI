"""Planner Agent: turns a free-text learning goal into a structured goal and a task plan.

Rules always work (mock mode). With an LLM the goal is parsed by the model and then validated; any failure falls
back to the rules. Privacy and academic-integrity guardrails are always enforced by rules, never by the LLM.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

from . import tools as T
from .tools import LEARNERS, TOPICS

ESC_RE = re.compile(r"\bhuman\b|real person|teacher|mentor|talk to (a |my )?(tutor|someone|person)|speak to", re.I)
INTEGRITY_RE = re.compile(r"write my|do my (homework|assignment)|answer key|exam answers|cheat", re.I)
LEARN_RE = re.compile(r"learn|teach|study|understand|master|improve|prepare|revise|practi[cs]e|help me with|explain", re.I)

DEFAULT_DAYS, DEFAULT_HPD = 7, 1.0


def mentions_other_learner(request, lid):
    r = request.lower()
    for oid, l in LEARNERS.items():
        if oid == lid:
            continue
        if oid.lower() in r or l["name"].lower() in r or re.search(rf"\b{re.escape(l['name'].split()[0].lower())}\b", r):
            return l["name"]
    return None


def parse_budget(request):
    """Days and hours per day from free text. Returns (days, hpd, assumptions)."""
    r = request.lower()
    days = hpd = None
    notes = []
    if m := re.search(r"(\d+)\s*weeks?", r):
        days = int(m.group(1)) * 7
    if m := re.search(r"(\d+)\s*days?\b", r):
        days = int(m.group(1))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", r):
        hpd = float(m.group(1))
    elif m := re.search(r"(\d+)\s*(?:minutes?|mins?)\b", r):
        hpd = int(m.group(1)) / 60
    if days is None:
        days = DEFAULT_DAYS
        notes.append(f"No time frame given: assumed {DEFAULT_DAYS} days")
    if hpd is None:
        hpd = DEFAULT_HPD
        notes.append(f"No daily time given: assumed {DEFAULT_HPD:g} hour per day")
    return clamp_budget(days, hpd, notes)


def clamp_budget(days, hpd, notes):
    d, h = min(max(int(days), 1), 60), min(max(float(hpd), 0.25), 8.0)
    if d != int(days) or h != float(hpd):
        notes.append(f"Budget adjusted to the allowed range: {d} days, {h:g} h/day")
    return d, h, notes


def rule_goal(request, lid):
    other = mentions_other_learner(request, lid)
    targets = T.find_topics(request)
    days, hpd, notes = parse_budget(request)
    if other:
        intent, why = "privacy", f"The request names another learner ({other}); their data is private."
    elif ESC_RE.search(request):
        intent, why = "escalate", "The learner asked for a human tutor."
    elif INTEGRITY_RE.search(request):
        intent, why = "out_of_scope", "I can't do assignments or exams for you, but I can help you learn the topic."
    elif not targets:
        intent, why = "out_of_scope", "No topic from the catalogue was recognised: " + ", ".join(t["name"] for t in TOPICS.values())
    else:
        intent, why = "learning_path", "A learning goal with recognised topics."
    return {"intent": intent, "rationale": why,
            "goal": {"targets": targets, "days": days, "hours_per_day": hpd, "assumptions": notes, "source": "rules"}}


class GoalOut(BaseModel):
    intent: Literal["learning_path", "escalate", "out_of_scope"]
    targets: list[str] = Field(default_factory=list, description="Topic ids the learner wants to master")
    days: int = Field(default=DEFAULT_DAYS, description="Number of days available")
    hours_per_day: float = Field(default=DEFAULT_HPD, description="Study hours available per day")


GOAL_PROMPT = """You are the Planner Agent of a personalized learning system for a machine learning course.
Parse the learner's request. Topic ids: {topics}. Use intent learning_path for learning goals, escalate if they ask for a
human tutor, out_of_scope if it is not about learning these topics. targets must be topic ids from the list.
If no time frame or daily time is given, use days=7 and hours_per_day=1."""


def llm_goal(helper, request):
    topics = "; ".join(f"{k} = {v['name']}" for k, v in TOPICS.items())
    out = helper.structured(GoalOut, GOAL_PROMPT.format(topics=topics), request)
    targets = [t.upper() for t in out.targets if t.upper() in TOPICS]
    days, hpd, notes = clamp_budget(out.days, out.hours_per_day, [])
    return {"intent": out.intent, "rationale": "Goal parsed by the LLM and validated",
            "goal": {"targets": targets, "days": days, "hours_per_day": hpd, "assumptions": notes, "source": "llm"}}


def make_goal(helper, request, lid):
    """Returns (result, notes). Guardrails (privacy, integrity) are always decided by rules first."""
    base = rule_goal(request, lid)
    notes = []
    guard = base["intent"] == "privacy" or bool(INTEGRITY_RE.search(request)) or bool(ESC_RE.search(request))
    if helper is None or guard:  # guardrails are decided by rules, never by the LLM
        return base, notes
    try:
        res = llm_goal(helper, request)
        if res["intent"] == "learning_path" and not res["goal"]["targets"]:
            if base["goal"]["targets"]:
                res["goal"]["targets"] = base["goal"]["targets"]
                notes.append("LLM found no topics; used the topics found by the rules")
            else:
                res["intent"], res["rationale"] = "out_of_scope", base["rationale"]
        return res, notes
    except Exception as e:
        notes.append(f"LLM goal parsing failed ({type(e).__name__}); used the rule-based parser")
        return base, notes


def make_tasks(goal, lid):
    t = goal["targets"]
    tasks = [
        ("diagnostic", "get_learner_profile", "Load the learner profile", {"learner_id": lid}),
        ("diagnostic", "compute_mastery_map", "Compute mastery per topic from quiz history", {"learner_id": lid}),
        ("diagnostic", "get_prerequisite_closure", "Find every prerequisite of the goal topics", {"topics": t}),
        ("diagnostic", "find_root_causes", "Separate root-cause gaps from downstream gaps", {}),
        ("tutor", "build_study_plan", "Schedule remediation within the time budget", {}),
        ("critic", "validate_plan", "Check order, budget, coverage, resources", {}),
        ("tutor", "teach_and_quiz", "Explain each topic and ask practice questions", {}),
        ("critic", "validate_quiz_and_assess", "Validate questions, grade answers, decide: cleared or re-teach", {}),
    ]
    return [{"step": i, "agent": a, "tool": n, "purpose": p, "args": g} for i, (a, n, p, g) in enumerate(tasks, 1)]

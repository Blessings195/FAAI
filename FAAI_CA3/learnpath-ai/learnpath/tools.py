"""Data access, shared helpers and LangChain tools for LearnPath AI. All data is dummy data in ../data."""
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

DATA = Path(__file__).resolve().parent.parent / "data"
TICKET_FILE = Path(os.getenv("LEARNPATH_TICKET_FILE", DATA / "tutor_tickets.jsonl"))

MASTERED = 0.7   # mastery at or above this counts as mastered
SHAKY = 0.5      # between SHAKY and MASTERED = shaky, below = gap
PRIOR = 0.4      # assumed mastery for topics with no quiz history (flagged "unassessed")


def _load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


TOPICS = {t["id"]: t for t in _load("topics.json")}
TOPIC_ORDER = list(TOPICS)
LEARNERS = {l["id"]: l for l in _load("learners.json")}
QUESTIONS = {q["id"]: q for q in _load("questions.json")}
RESOURCES = {r["id"]: r for r in _load("resources.json")}

ALIASES = {
    "PY": ["python", "numpy", "pandas"], "LA": ["linear algebra", "vectors", "matrices", "matrix"],
    "CALC": ["calculus", "derivative", "derivatives"], "PROB": ["probability"],
    "PREP": ["preprocessing", "data cleaning", "feature scaling"], "LINREG": ["linear regression"],
    "GD": ["gradient descent"], "LOGREG": ["logistic regression"],
    "EVAL": ["model evaluation", "evaluation metrics", "cross-validation", "cross validation", "precision and recall"],
    "REG": ["regularization", "regularisation", "overfitting"], "NN": ["neural network", "neural networks"],
    "BP": ["backpropagation", "backprop", "back-propagation", "back propagation"],
}


# ------------------------------------------------------------------ helpers
def round_half(x):
    """Round to the nearest 0.5 (half up), robust to float noise."""
    return math.floor(x * 2 + 0.5 + 1e-9) / 2


def mastery_from_attempts(attempts):
    """Recency-weighted accuracy: attempt i (1 = oldest) has weight i. None if there are no attempts."""
    if not attempts:
        return None
    num = sum(i * (a["score"] / a["max"]) for i, a in enumerate(attempts, 1))
    den = sum(range(1, len(attempts) + 1))
    return num / den


def status_of(mastery, assessed):
    if not assessed:
        return "unassessed"
    return "mastered" if mastery >= MASTERED - 1e-9 else "shaky" if mastery >= SHAKY - 1e-9 else "gap"


def get_learner(learner_id):
    l = LEARNERS.get(str(learner_id).strip().upper())
    if l is None:
        raise KeyError(f"No learner with ID '{learner_id}'")
    return l


def mastery_map(learner_id, extra=None):
    """Mastery per topic from the stored history. Quiz attempts made during this session (extra) overlay it:
    the latest session quiz counts double, because the learner has studied since the older attempts."""
    l = get_learner(learner_id)
    hist = {t: [] for t in TOPICS}
    ses = {t: [] for t in TOPICS}
    for a in l["history"]:
        hist[a["topic"]].append(a)
    for a in extra or []:
        ses[a["topic"]].append(a)
    out = {}
    for t in TOPICS:
        h = mastery_from_attempts(hist[t])
        if ses[t]:
            last = ses[t][-1]
            r = last["score"] / last["max"]
            value, assessed = (r if h is None else (h + 2 * r) / 3), True
        else:
            value, assessed = (PRIOR if h is None else h), h is not None
        out[t] = {"mastery": round(value, 3), "status": status_of(value, assessed),
                  "attempts": len(hist[t]) + len(ses[t]), "assessed": assessed}
    return out


def prerequisite_closure(targets):
    """Targets plus every transitive prerequisite, in topological order (prerequisites first)."""
    seen, order = set(), []

    def visit(t):
        if t in seen:
            return
        seen.add(t)
        for p in sorted(TOPICS[t]["prerequisites"], key=TOPIC_ORDER.index):
            visit(p)
        order.append(t)

    for t in targets:
        visit(t)
    return order


def dependents(topic):
    return [t for t in TOPIC_ORDER if topic in TOPICS[t]["prerequisites"]]


def find_topics(text):
    """Topic ids mentioned in free text, in order of appearance."""
    r = text.lower()
    hits = []
    for tid, words in ALIASES.items():
        names = words + [TOPICS[tid]["name"].lower()]
        pos = [m.start() for w in names for m in re.finditer(rf"\b{re.escape(w)}\b", r)]
        if pos:
            hits.append((min(pos), tid))
    return [tid for _, tid in sorted(hits)]


# ------------------------------------------------------------------ tools
@tool
def get_learner_profile(learner_id: str) -> dict:
    """Get a learner's profile: name, level and how much quiz history exists."""
    try:
        l = get_learner(learner_id)
    except KeyError as e:
        return {"error": str(e)}
    return {"learner_id": l["id"], "name": l["name"], "level": l["level"],
            "quiz_attempts": len(l["history"]), "topics_assessed": len({a["topic"] for a in l["history"]})}


@tool
def compute_mastery_map(learner_id: str) -> dict:
    """Compute mastery (0 to 1) and status (mastered, shaky, gap, unassessed) for every topic from quiz history."""
    try:
        return {"mastery": mastery_map(learner_id)}
    except KeyError as e:
        return {"error": str(e)}


@tool
def get_prerequisite_closure(topics: list[str]) -> dict:
    """Return the given topics plus all their prerequisites in learning order, and the prerequisite edges."""
    topics = [t.upper() for t in topics if t.upper() in TOPICS]
    order = prerequisite_closure(topics)
    return {"order": order, "edges": [[p, t] for t in order for p in TOPICS[t]["prerequisites"]]}


@tool
def find_resources(topic: str, level: str = "") -> dict:
    """Find study resources for a topic, optionally filtered by level (beginner or intermediate)."""
    topic = topic.upper()
    res = [r for r in RESOURCES.values() if r["topic"] == topic and (not level or r["level"] == level)]
    return {"resources": res}


@tool
def get_practice_questions(topic: str, difficulties: list[int], exclude_ids: list[str], prefer_ids: list[str]) -> dict:
    """Pick practice questions from the bank for a topic. difficulties gives the wanted difficulty (1 to 3) per
    question; exclude_ids are questions already seen; prefer_ids are previously missed questions to ask again."""
    topic = topic.upper()
    pool = [q for q in QUESTIONS.values() if q["topic"] == topic]
    lo, hi = max(1, min(difficulties) - 1), max(difficulties)
    chosen = []
    for pid in prefer_ids[:1]:
        if pid in QUESTIONS and QUESTIONS[pid]["topic"] == topic:
            chosen.append(QUESTIONS[pid])
    for d in difficulties[len(chosen):]:
        free = [q for q in pool if q not in chosen]
        in_range = [q for q in free if lo <= q["difficulty"] <= hi]
        tiers = [[q for q in in_range if q["id"] not in exclude_ids],                       # fresh, right difficulty
                 [q for q in in_range if q["id"] in prefer_ids],                            # missed before, right difficulty
                 in_range,                                                                  # repeat an easier one rather than a too-hard one
                 [q for q in free if q["id"] not in exclude_ids or q["id"] in prefer_ids],  # fresh or missed, any difficulty
                 free]
        cands = next((c for c in tiers if c), [])
        if cands:
            chosen.append(min(cands, key=lambda q: (abs(q["difficulty"] - d), q["id"])))
    chosen.sort(key=lambda q: (q["difficulty"], q["id"]))
    return {"questions": [dict(q) for q in chosen]}


@tool
def escalate_to_tutor(learner_id: str, topic: str, reason: str) -> dict:
    """Create an escalation ticket for a human tutor when a learner is stuck or needs human help."""
    ticket = {"ticket_id": "TUT-" + uuid.uuid4().hex[:6].upper(), "learner_id": str(learner_id).upper(),
              "topic": topic, "reason": reason, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "status": "OPEN"}
    with open(TICKET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ticket) + "\n")
    return {"ticket_id": ticket["ticket_id"], "status": "OPEN", "message": "A human tutor will follow up."}

"""Tutor Agent logic: builds the study plan, writes explanations and chooses practice questions."""
import copy
import math

from . import tools as T
from .diagnosis import required_work
from .llm import parse_json
from .tools import QUESTIONS, TOPICS

ACTIVITY = {"video": "Watch", "article": "Read", "exercise": "Practice"}
FAKE_RESOURCE = "RES-FAKE-9"  # used only by the fault-injection demo


# ------------------------------------------------------------------ study plan
def pick_resources(topic, mastery_value, kind):
    res = T.find_resources.invoke({"topic": topic})["resources"]
    beginner = [r for r in res if r["level"] == "beginner"]
    inter = [r for r in res if r["level"] == "intermediate"]
    if kind == "challenge":
        return inter[::-1]
    if mastery_value < 0.5:
        return beginner + inter[:1]
    return inter[:1] + beginner[1:2] + inter[1:]


def build_plan(diag, goal, mastery, fault=False):
    days, hpd = goal["days"], goal["hours_per_day"]
    budget_min, cap = int(round(days * hpd * 60)), int(round(hpd * 60))
    work = required_work(diag, goal, mastery)
    remaining, items, deferred = budget_min, [], []
    for w in work:
        want = int(round(w["hours_needed"] * 60))
        give = min(want, remaining)
        give = (give // 15) * 15
        if give < 30:
            give = 0
        item = dict(w, minutes=give, partial=0 < give < want, deferred=give == 0)
        if give == 0:
            deferred.append({"topic": w["topic"], "reason": "not enough time in the budget"})
        remaining -= give
        items.append(item)

    sessions, day, used = [], 1, 0
    for it in items:
        if it["deferred"]:
            continue
        seq = pick_resources(it["topic"], it["mastery"], it["kind"])
        left, i = it["minutes"], 0
        while left > 0:
            chunk = min(left, cap - used, 60)
            r = seq[i % len(seq)]
            sessions.append({"day": day, "topic": it["topic"], "minutes": chunk, "resource_id": r["id"],
                             "activity": ACTIVITY[r["type"]]})
            left, used, i = left - chunk, used + chunk, i + 1
            if used >= cap:
                day, used = day + 1, 0

    needed = round(sum(w["hours_needed"] for w in work), 2)
    allocated = round(sum(it["minutes"] for it in items) / 60, 2)
    budget_h = round(budget_min / 60, 2)
    plan = {"budget_hours": budget_h, "needed_hours": needed, "allocated_hours": allocated,
            "shortfall_hours": round(max(0.0, needed - budget_h), 2), "feasible": needed <= budget_h + 1e-9,
            "days_needed": math.ceil(needed / hpd - 1e-9), "items": items, "sessions": sessions,
            "deferred": deferred, "mode": diag["mode"]}
    plan["order"] = [it["topic"] for it in items if not it["deferred"]]
    if fault:
        corrupt_plan(plan, mastery)
    return plan


def corrupt_plan(plan, mastery):
    """Fault injection: make a plan that breaks several rules so the Critic can be demonstrated."""
    mastered = [t for t, v in mastery.items() if v["status"] == "mastered" and t not in plan["order"]]
    items = plan["items"]
    scheduled = [i for i in items if not i["deferred"]]
    if scheduled and len(scheduled) > 1:  # put the last topic first: it now precedes its prerequisites
        last = scheduled[-1]
        items.remove(last)
        items.insert(0, last)
    if mastered:  # waste time on a topic the learner already mastered
        t = mastered[0]
        items.insert(0, {"topic": t, "kind": "remediate", "role": "prerequisite", "mastery": mastery[t]["mastery"],
                         "hours_needed": 1.0, "minutes": 60, "partial": False, "deferred": False})
        plan["sessions"].insert(0, {"day": 1, "topic": t, "minutes": 60, "resource_id": f"RES-{t}-1", "activity": "Watch"})
    plan["needed_hours"] = round(plan["needed_hours"] + 1.5, 2)  # wrong number in the summary
    if plan["sessions"]:
        plan["sessions"][-1]["resource_id"] = FAKE_RESOURCE  # a resource that does not exist
    plan["order"] = [i["topic"] for i in items if not i["deferred"]]


# ------------------------------------------------------------------ teaching
def _correct_text(q):
    return q["options"][q["answer"]]


def template_explanation(first_name, topic, entry, round_no, missed):
    t = TOPICS[topic]
    pct = round(entry["mastery"] * 100)
    if round_no == 1:
        lines = [f"**{t['name']}**: {t['summary']}", "", "Key points:"] + [f"- {k}" for k in t["key_points"]]
        lines += ["", f"Where you stand, {first_name}: mastery {pct}% ({entry['status']}).",
                  f"Watch out: {t['common_mistake']}"]
    else:
        lines = [f"Let's look at **{t['name']}** another way, {first_name}.", f"Analogy: {t['analogy']}", ""]
        if missed:
            lines.append("What went wrong last time:")
            lines += [f"- {q['text']} Correct answer: {_correct_text(q)}. {q['explanation']}" for q in missed]
            lines.append("")
        lines.append(f"Watch out: {t['common_mistake']}")
    return "\n".join(lines)


EXPLAIN_PROMPT = """You are a patient, encouraging tutor for a machine learning course. Explain the topic to the learner in at most
140 words using ONLY the facts provided. Use plain language, address the learner by first name, and finish with one practical
tip. If this is a re-teach, use the analogy and the mistakes the learner made. Do not invent formulas."""


def explain(helper, first_name, topic, entry, round_no, missed):
    """Returns (text, source, note). LLM output is guard-checked; any problem falls back to the template."""
    fallback = template_explanation(first_name, topic, entry, round_no, missed)
    if helper is None:
        return fallback, "template", ""
    t = TOPICS[topic]
    facts = {"learner": first_name, "topic": t["name"], "summary": t["summary"], "key_points": t["key_points"],
             "analogy": t["analogy"], "common_mistake": t["common_mistake"], "mastery_percent": round(entry["mastery"] * 100),
             "round": round_no, "missed": [{"question": q["text"], "correct": _correct_text(q), "why": q["explanation"]} for q in missed]}
    try:
        import json
        text = helper.text(EXPLAIN_PROMPT, json.dumps(facts))
    except Exception as e:
        return fallback, "template", f"LLM explanation failed ({type(e).__name__}); used the template"
    if not (30 <= len(text) <= 1400) or t["name"].lower() not in text.lower():
        return fallback, "template", "LLM explanation failed the guard check (length or off-topic); used the template"
    return text, "llm", ""


# ------------------------------------------------------------------ quizzes
def quiz_difficulties(mastery_value, kind):
    if kind == "challenge":
        return [2, 2, 3]
    if mastery_value < 0.4:
        return [1, 1, 2]
    if mastery_value < 0.7:
        return [1, 2, 2]
    return [2, 2, 3]


def bank_quiz(topic, difficulties, seen, missed):
    qs = T.get_practice_questions.invoke({"topic": topic, "difficulties": difficulties,
                                          "exclude_ids": list(seen), "prefer_ids": list(missed)})["questions"]
    return qs


GEN_PROMPT = """You write multiple-choice questions for a machine learning course. Return a JSON array only. Each item:
{"text": str, "options": [4 distinct strings], "answer": index of the correct option (0-3), "explanation": str,
"difficulty": 1-3}. Exactly one option must be correct. Questions must be about the given topic only."""


def llm_quiz(helper, topic, difficulties, avoid_texts):
    import json
    t = TOPICS[topic]
    user = json.dumps({"topic": t["name"], "facts": [t["summary"]] + t["key_points"], "count": len(difficulties),
                       "difficulties": difficulties, "do_not_repeat": avoid_texts})
    raw = helper.text(GEN_PROMPT, user)
    items = parse_json(raw)
    if isinstance(items, dict):
        items = items.get("questions", [items])
    out = []
    for k, it in enumerate(items[:len(difficulties)], 1):
        out.append({"id": f"GEN-{topic}-{k}", "topic": topic, "difficulty": int(it.get("difficulty", difficulties[min(k - 1, len(difficulties) - 1)])),
                    "text": str(it.get("text", "")), "options": [str(o) for o in it.get("options", [])],
                    "answer": it.get("answer"), "explanation": str(it.get("explanation", "")), "generated": True})
    return out


def corrupt_quiz(questions):
    """Fault injection: invalid answer key on one question and the wrong topic on another."""
    qs = copy.deepcopy(questions)
    if qs:
        qs[0]["answer"] = 9
    if len(qs) > 1:
        other = next(t for t in TOPICS if t != qs[1]["topic"])
        qs[1]["topic"] = other
    return qs


def public_questions(questions):
    """What the learner sees: no answer key and no explanation."""
    return [{"id": q["id"], "text": q["text"], "options": q["options"], "difficulty": q["difficulty"]} for q in questions]

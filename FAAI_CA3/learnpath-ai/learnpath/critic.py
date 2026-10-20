"""Critic Agent: validates study plans and practice quizzes against freshly queried source data."""
import re

from . import tools as T
from .diagnosis import diagnose, required_work
from .tools import RESOURCES, TOPICS

VERIFY_PROMPT = ("Answer the multiple-choice question. Reply with only the number of the correct option "
                 "(0-based index).")


def _result(checks, attempt):
    issues = [f"{c['check']}: {c['detail']}" for c in checks if not c["passed"]]
    return {"attempt": attempt, "passed": not issues, "checks": checks, "issues": issues}


def check_plan(learner_id, goal, plan, attempt):
    """Re-derive mastery and the required work from the raw data, then check the plan against it."""
    mastery = T.compute_mastery_map.invoke({"learner_id": learner_id})["mastery"]
    diag = diagnose(mastery, goal["targets"])
    work = {w["topic"]: w for w in required_work(diag, goal, mastery)}
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    items = plan["items"]
    scheduled = [i for i in items if not i.get("deferred")]
    order = [i["topic"] for i in scheduled]
    days, hpd = goal["days"], goal["hours_per_day"]
    budget_min, cap = int(round(days * hpd * 60)), int(round(hpd * 60))
    need_total = round(sum(w["hours_needed"] for w in work.values()), 2)

    add("Total hours needed matches an independent recomputation", abs(plan["needed_hours"] - need_total) < 0.01,
        f"plan says {plan['needed_hours']} h, recomputed {need_total} h")

    bad = []
    for pos, t in enumerate(order):
        for p in TOPICS[t]["prerequisites"]:
            if mastery[p]["status"] != "mastered" and (p not in order or order.index(p) > pos):
                bad.append(f"{t} before {p}" if p in order else f"{t} without {p}")
    add("Prerequisites are scheduled before the topics that need them", not bad,
        "Order is valid." if not bad else "Order problems: " + ", ".join(bad))

    total = sum(s["minutes"] for s in plan["sessions"])
    per_day = {}
    for s in plan["sessions"]:
        per_day[s["day"]] = per_day.get(s["day"], 0) + s["minutes"]
    over_days = [d for d, m in per_day.items() if m > cap or d > days]
    add("Time budget respected (total and per day)", total <= budget_min and not over_days,
        f"{total} of {budget_min} minutes used" + (f"; over the daily limit on day {over_days}" if over_days else ""))

    wasted = [i["topic"] for i in scheduled if i["kind"] == "remediate" and mastery[i["topic"]]["status"] == "mastered"]
    add("No time spent on topics the learner has already mastered", not wasted,
        "None." if not wasted else "Already mastered: " + ", ".join(wasted))

    deferred_topics = {d["topic"] for d in plan.get("deferred", [])}
    missing = [r for r in diag["roots"] if r not in order and r not in deferred_topics]
    add("Every root-cause gap is covered or openly deferred", not missing,
        "Covered." if not missing else "Missing: " + ", ".join(missing))

    problems = []
    for s in plan["sessions"]:
        r = RESOURCES.get(s["resource_id"])
        if r is None:
            problems.append(f"{s['resource_id']} does not exist")
        elif r["topic"] != s["topic"]:
            problems.append(f"{s['resource_id']} is not about {s['topic']}")
    for i in scheduled:
        first = next((s for s in plan["sessions"] if s["topic"] == i["topic"]), None)
        if first and i["kind"] == "remediate" and mastery[i["topic"]]["mastery"] < 0.5:
            r = RESOURCES.get(first["resource_id"])
            if r and r["level"] != "beginner":
                problems.append(f"{i['topic']} starts above beginner level")
    add("Resources exist, match the topic and suit the learner's level", not problems,
        "All resources valid." if not problems else "; ".join(problems))

    feasible = need_total <= round(budget_min / 60, 2) + 1e-9
    short = round(max(0.0, need_total - budget_min / 60), 2)
    add("Feasibility is disclosed honestly", plan["feasible"] == feasible and abs(plan["shortfall_hours"] - short) < 0.01,
        f"plan: feasible={plan['feasible']}, shortfall={plan['shortfall_hours']} h; recomputed: feasible={feasible}, shortfall={short} h")

    off = [i["topic"] for i in scheduled if i["topic"] not in work or
           (not i.get("partial") and abs(i["minutes"] / 60 - work[i["topic"]]["hours_needed"]) > 0.01) or
           (i.get("partial") and i["minutes"] / 60 > work[i["topic"]]["hours_needed"] + 0.01)]
    add("Hours per topic match the diagnosis", not off, "All match." if not off else "Mismatch: " + ", ".join(off))
    return _result(checks, attempt)


def verify_with_llm(helper, q):
    """Independent self-consistency check for an LLM-written question: can the model answer it the same way?"""
    prompt = q["text"] + "\n" + "\n".join(f"{i}: {o}" for i, o in enumerate(q["options"]))
    try:
        reply = helper.text(VERIFY_PROMPT, prompt)
        m = re.search(r"\d", reply)
        return m is not None and int(m.group()) == q["answer"]
    except Exception:
        return False


def check_quiz(topic, quiz, wanted, seen_ids, missed_ids, attempt, helper=None):
    qs = quiz["questions"]
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    bad = []
    for q in qs:
        opts, ans = q.get("options"), q.get("answer")
        if not isinstance(q.get("text"), str) or not q["text"].strip():
            bad.append(f"{q.get('id')}: empty text")
        elif not isinstance(opts, list) or not 2 <= len(opts) <= 5 or len(set(opts)) != len(opts):
            bad.append(f"{q['id']}: options must be 2 to 5 distinct strings")
        elif not isinstance(ans, int) or isinstance(ans, bool) or not 0 <= ans < len(opts):
            bad.append(f"{q['id']}: answer key {ans!r} is not a valid option")
    add("Question format is valid (text, distinct options, valid answer key)", not bad,
        "All questions valid." if not bad else "; ".join(bad))

    off = [q["id"] for q in qs if q.get("topic") != topic]
    add("Questions belong to the requested topic", not off, "All on topic." if not off else "Wrong topic: " + ", ".join(off))

    diffs = [q.get("difficulty") for q in qs]
    ok = len(qs) == len(wanted) and all(isinstance(d, int) and min(wanted) - 1 <= d <= max(wanted) for d in diffs)
    add("Question count and difficulty match the request", ok, f"{len(qs)} questions, difficulty {diffs}; wanted {wanted}")

    texts = [re.sub(r"\W+", " ", str(q.get("text", ""))).strip().lower() for q in qs]
    dup = len({q.get("id") for q in qs}) != len(qs) or len(set(texts)) != len(texts)
    add("No duplicate questions", not dup, "No duplicates." if not dup else "Duplicate question found")

    noexp = [q.get("id") for q in qs if not str(q.get("explanation", "")).strip()]
    add("Every question has an explanation", not noexp, "All have explanations." if not noexp else "Missing: " + ", ".join(map(str, noexp)))

    lo, hi = max(1, min(wanted) - 1), max(wanted)
    used = {q["id"] for q in qs}
    fresh_left = [q for q in T.QUESTIONS.values() if q["topic"] == topic and q["id"] not in seen_ids
                  and q["id"] not in used and lo <= q["difficulty"] <= hi]
    stale = [q["id"] for q in qs if q["id"] in seen_ids and q["id"] not in missed_ids]
    bad_repeat = stale if fresh_left else []  # repeating a correct question is fine only when no fresh one fits
    add("Only unseen questions, or ones the learner missed before", not bad_repeat,
        "Fresh, or repeats were unavoidable." if not bad_repeat else "Already answered correctly: " + ", ".join(bad_repeat))

    gen = [q for q in qs if q.get("generated")]
    if gen:
        if helper is None:
            add("LLM-written questions pass an independent self-check", False, "No LLM available to verify them")
        else:
            failed = [q["id"] for q in gen if not (not bad and verify_with_llm(helper, q))]
            add("LLM-written questions pass an independent self-check", not failed,
                "The model reproduced every answer key." if not failed else "Disagrees with the key: " + ", ".join(failed))
    return _result(checks, attempt)

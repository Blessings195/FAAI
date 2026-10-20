"""Diagnostic Agent logic: knowledge-gap analysis over the prerequisite graph."""
from . import tools as T
from .tools import TOPICS, TOPIC_ORDER


def diagnose(mastery, targets):
    """Find which topics need work and which gaps are root causes.

    A topic needs work if it is not mastered. It is a *root cause* if none of its prerequisites need work; the other
    gaps are *downstream* (blocked until their prerequisites are fixed). If nothing needs work the mode is
    "challenge" and the next unlocked topic is suggested as a stretch goal.
    """
    closure = T.prerequisite_closure(targets)
    need_work = [t for t in closure if mastery[t]["status"] != "mastered"]
    blocked_by = {t: [p for p in TOPICS[t]["prerequisites"] if p in need_work] for t in need_work}
    roots = [t for t in need_work if not blocked_by[t]]
    hours = {t: max(0.5, T.round_half(TOPICS[t]["est_hours"] * (1 - mastery[t]["mastery"]))) for t in need_work}

    order, remaining = [], set(need_work)  # prerequisites first; weakest topic first among equals
    while remaining:
        avail = [t for t in remaining if not any(p in remaining for p in TOPICS[t]["prerequisites"])]
        avail.sort(key=lambda t: (mastery[t]["mastery"], TOPIC_ORDER.index(t)))
        order.append(avail[0])
        remaining.remove(avail[0])

    stretch = None
    if not need_work:
        for t in targets:
            for d in T.dependents(t):
                ready = all(mastery[p]["status"] == "mastered" for p in TOPICS[d]["prerequisites"])
                if mastery[d]["status"] != "mastered" and ready and d not in targets:
                    stretch = d
                    break
            if stretch:
                break
    roles = {}
    for t in closure:
        roles[t] = ("mastered" if t not in need_work else "root cause" if t in roots else "downstream")
    chains = [f"{t} is blocked by {', '.join(blocked_by[t])}" for t in need_work if blocked_by[t]]
    return {"targets": list(targets), "closure": closure, "need_work": need_work, "roots": roots, "blocked_by": blocked_by,
            "needed_hours": hours, "order": order, "roles": roles, "chains": chains, "stretch": stretch,
            "mode": "remediate" if need_work else "challenge",
            "unassessed": [t for t in closure if not mastery[t]["assessed"]]}


def required_work(diag, goal, mastery):
    """The work the plan must cover, with hours needed per item (shared by the Tutor and the Critic)."""
    items = []
    if diag["mode"] == "remediate":
        for t in diag["order"]:
            role = "target" if t in goal["targets"] else "root cause" if t in diag["roots"] else "prerequisite"
            items.append({"topic": t, "kind": "remediate", "role": role, "mastery": mastery[t]["mastery"],
                          "hours_needed": diag["needed_hours"][t]})
    else:
        for t in goal["targets"]:
            items.append({"topic": t, "kind": "challenge", "role": "target", "mastery": mastery[t]["mastery"],
                          "hours_needed": 1.0})
        s = diag["stretch"]
        if s:
            h = max(0.5, T.round_half(TOPICS[s]["est_hours"] * (1 - mastery[s]["mastery"])))
            items.append({"topic": s, "kind": "stretch", "role": "stretch", "mastery": mastery[s]["mastery"],
                          "hours_needed": h})
    return items

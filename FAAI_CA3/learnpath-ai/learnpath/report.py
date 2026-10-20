"""Builds the learner-facing report (Markdown) and the before/after progress table."""
from . import tools as T
from .tools import TOPICS


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def pct(x):
    return f"{round(x * 100)}%"


def progress_rows(state):
    """Topic, role, mastery before, mastery now for every topic in the goal's prerequisite closure."""
    live = T.mastery_map(state["learner_id"], state.get("history_add", []))
    base, diag = state["mastery_base"], state["diagnosis"]
    return [{"topic": t, "name": TOPICS[t]["name"], "role": diag["roles"][t],
             "before": base[t]["mastery"], "after": live[t]["mastery"], "status": live[t]["status"]}
            for t in diag["closure"]]


def day_table(plan):
    rows = []
    for s in plan["sessions"]:
        rows.append((s["day"], TOPICS[s["topic"]]["name"], s["activity"], T.RESOURCES.get(s["resource_id"], {}).get("title", s["resource_id"]), s["minutes"]))
    return sorted(rows, key=lambda r: r[0])


def topic_outcomes(state):
    by = {}
    for r in state.get("results", []):
        by.setdefault(r["topic"], []).append(r)
    tickets = {t["topic"]: t["ticket_id"] for t in state.get("tickets", [])}
    out = []
    for topic, rs in by.items():
        last = rs[-1]
        outcome = "Cleared" if last["cleared"] else f"Escalated to a human tutor ({tickets.get(topic, 'ticket')})" \
            if last["decision"] == "escalate" else "Needs another round"
        out.append({"topic": topic, "rounds": len(rs), "scores": ", ".join(f"{r['correct']}/{r['total']}" for r in rs),
                    "outcome": outcome, "before": rs[0]["mastery_before"], "after": last["mastery_after"]})
    return out


def build_report(state):
    lid = state["learner_id"]
    name = T.get_learner(lid)["name"]
    goal, diag, plan = state["goal"], state["diagnosis"], state["plan"]
    L = [f"# LearnPath report for {name}", ""]
    L.append(f"**Goal:** {', '.join(TOPICS[t]['name'] for t in goal['targets'])} in {goal['days']} days at "
             f"{goal['hours_per_day']:g} h/day (budget {plan['budget_hours']:g} h).")
    for a in goal["assumptions"]:
        L.append(f"- _Assumption: {a}_")
    L += ["", "## Where you started", ""]
    rows = progress_rows(state)
    L.append(md_table(["Topic", "Role in your goal", "Mastery at start", "Status now"],
                      [(r["name"], r["role"], pct(r["before"]), r["status"]) for r in rows]))
    if diag["mode"] == "remediate":
        L.append("")
        L.append(f"**Root causes to fix first:** {', '.join(diag['roots'])}. "
                 + ("Other gaps are downstream: " + "; ".join(diag["chains"]) + "." if diag["chains"] else ""))
    else:
        L.append("")
        L.append("**No gaps found.** You already master the goal topics and everything they depend on, so the plan "
                 "uses harder challenge questions and previews the next topic"
                 + (f" ({TOPICS[diag['stretch']]['name']})." if diag["stretch"] else "."))
    L += ["", "## Your study path", ""]
    if not plan["feasible"]:
        L.append(f"> **Time warning:** you need {plan['needed_hours']:g} h but have {plan['budget_hours']:g} h "
                 f"(short by {plan['shortfall_hours']:g} h). Extend to **{plan['days_needed']} days** at this pace to cover everything. "
                 "Until then the plan fixes the root causes first and defers the rest.")
        L.append("")
    L.append(md_table(["Day", "Topic", "Activity", "Resource", "Minutes"], day_table(plan)))
    if plan["deferred"]:
        L.append("")
        L.append("**Deferred:** " + ", ".join(f"{TOPICS[d['topic']]['name']} ({d['reason']})" for d in plan["deferred"]) + ".")
    partial = [i["topic"] for i in plan["items"] if i.get("partial")]
    if partial:
        L.append("")
        L.append("**Partly covered:** " + ", ".join(TOPICS[t]["name"] for t in partial) + " (less time than needed).")
    out = topic_outcomes(state)
    L += ["", "## Practice results", ""]
    if out:
        L.append(md_table(["Topic", "Rounds", "Scores", "Outcome", "Mastery"],
                          [(TOPICS[o["topic"]]["name"], o["rounds"], o["scores"], o["outcome"], f"{pct(o['before'])} → {pct(o['after'])}") for o in out]))
    else:
        L.append("No practice session has been completed yet.")
    L += ["", "## Next steps", ""]
    steps = []
    for t in state.get("tickets", []):
        steps.append(f"Meet your human tutor about **{TOPICS[t['topic']]['name']}** (ticket {t['ticket_id']}).")
    for d in plan["deferred"]:
        steps.append(f"Schedule **{TOPICS[d['topic']]['name']}** next: it was deferred for lack of time.")
    pending = [q["topic"] for q in state.get("queue", [])] + ([state["current"]["topic"]] if state.get("current") and state.get("stopped") else [])
    for t in pending:
        steps.append(f"Finish practising **{TOPICS[t]['name']}**.")
    if diag["mode"] == "challenge" and diag.get("stretch") and not any(r["topic"] == diag["stretch"] for r in out):
        steps.append(f"Try the stretch topic **{TOPICS[diag['stretch']]['name']}**.")
    if not steps:
        steps.append("All planned topics are cleared. Re-run the diagnosis after a few days to check retention.")
    L += [f"- {s}" for s in steps]
    c = state.get("critiques", [])
    caught = sum(1 for x in c if not x["passed"])
    L += ["", "## Validation", "",
          f"The Critic ran {len(c)} validation round(s) on the plan and the practice questions and rejected {caught} "
          f"draft(s) before they reached you."]
    return "\n".join(L)

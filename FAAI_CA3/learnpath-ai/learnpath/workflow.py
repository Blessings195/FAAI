"""LangGraph orchestration for LearnPath AI.

planner -> diagnose -> tutor_plan -> critic_plan --(retry)--> tutor_plan
                                          |-> start_practice -> next_topic -> teach -> make_quiz -> critic_quiz -> ask
ask (waits for the learner) -> assess -> cleared: next_topic | re-teach: teach | stuck: escalate_topic -> next_topic
next_topic (queue empty) -> finalize.   Side exits: refuse, escalate.
"""
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import assessor as AS
from . import critic as C
from . import diagnosis as D
from . import planner as P
from . import report as R
from . import tools as T
from . import tutor as U
from .tools import QUESTIONS, TOPICS

MAX_PLAN_ATTEMPTS = 2
MAX_QUIZ_ATTEMPTS = 3

REFUSALS = {"privacy": "I can only work with your own learning data, so I can't show or use another learner's information."}


class State(TypedDict, total=False):
    request: str
    learner_id: str
    inject_fault: bool
    intent: str
    rationale: str
    goal: dict
    tasks: list
    profile: dict
    mastery_base: dict
    diagnosis: dict
    plan: dict
    plan_attempts: int
    queue: list
    current: dict
    quiz: dict
    quiz_attempts: int
    quiz_fault_done: bool
    answers: dict
    stopped: bool
    decision: str
    history_add: list
    results: list
    tickets: list
    status: str
    final: str
    trace: Annotated[list, add]
    critiques: Annotated[list, add]


def log(agent, action, detail="", data=None):
    return {"agent": agent, "action": action, "detail": detail, "data": data}


def build_workflow(helper=None, use_llm_goal=True, use_llm_explain=True, use_llm_questions=False, checkpointer=None):
    def planner(state: State):
        lid = state["learner_id"]
        res, notes = P.make_goal(helper if use_llm_goal else None, state["request"], lid)
        goal, intent = res["goal"], res["intent"]
        trace = [log("Planner", f"Intent: {intent}", f"{res['rationale']} (goal parsed by {goal['source']})")]
        trace += [log("Planner", "Fallback", n) for n in notes]
        tasks = []
        if intent == "learning_path":
            names = ", ".join(TOPICS[t]["name"] for t in goal["targets"])
            trace.append(log("Planner", "Goal", f"{names}; {goal['days']} days x {goal['hours_per_day']:g} h = "
                             f"{goal['days'] * goal['hours_per_day']:g} h budget", goal))
            trace += [log("Planner", "Assumption", a) for a in goal["assumptions"]]
            tasks = P.make_tasks(goal, lid)
            trace.append(log("Planner", f"Task plan ({len(tasks)} tasks)", " → ".join(t["tool"] for t in tasks), tasks))
        return {"intent": intent, "rationale": res["rationale"], "goal": goal, "tasks": tasks, "plan_attempts": 0,
                "history_add": [], "results": [], "tickets": [], "trace": trace}

    def diagnose(state: State):
        lid, goal = state["learner_id"], state["goal"]
        prof = T.get_learner_profile.invoke({"learner_id": lid})
        mm = T.compute_mastery_map.invoke({"learner_id": lid})["mastery"]
        clo = T.get_prerequisite_closure.invoke({"topics": goal["targets"]})
        diag = D.diagnose(mm, goal["targets"])
        trace = [log("Diagnostic", "Called get_learner_profile", f"{prof['name']}, {prof['quiz_attempts']} past quiz attempts", prof),
                 log("Diagnostic", "Called compute_mastery_map", f"{sum(1 for v in mm.values() if v['status'] == 'mastered')} of {len(mm)} topics mastered", mm),
                 log("Diagnostic", "Called get_prerequisite_closure", " → ".join(clo["order"]), clo)]
        if diag["mode"] == "remediate":
            trace.append(log("Diagnostic", "Found knowledge gaps",
                             f"root causes: {', '.join(diag['roots'])}; also needs work: {', '.join(t for t in diag['need_work'] if t not in diag['roots']) or 'none'}", diag))
        else:
            trace.append(log("Diagnostic", "No gaps in the goal topics", f"challenge mode; stretch topic: {diag['stretch'] or 'none'}", diag))
        return {"profile": prof, "mastery_base": mm, "diagnosis": diag, "trace": trace}

    def tutor_plan(state: State):
        attempts = state.get("plan_attempts", 0)
        fault = bool(state.get("inject_fault")) and attempts == 0
        trace = []
        if attempts > 0:
            trace.append(log("Tutor", "Rebuilding the plan after Critic feedback", "; ".join(state["critiques"][-1]["issues"])))
        plan = U.build_plan(state["diagnosis"], state["goal"], state["mastery_base"], fault=fault)
        detail = "Deliberately corrupted plan (fault injection demo)" if fault else \
            f"{len(plan['order'])} topics, {plan['allocated_hours']:g} of {plan['budget_hours']:g} h used" + ("" if plan["feasible"] else f"; short by {plan['shortfall_hours']:g} h")
        trace.append(log("Tutor", f"Built study plan (attempt {attempts + 1})", detail, plan))
        return {"plan": plan, "plan_attempts": attempts + 1, "trace": trace}

    def critic_plan(state: State):
        c = C.check_plan(state["learner_id"], state["goal"], state["plan"], state["plan_attempts"])
        ok = sum(x["passed"] for x in c["checks"])
        verdict = "PASSED" if c["passed"] else "FAILED"
        detail = f"{ok}/{len(c['checks'])} checks passed" + ("" if c["passed"] else ": " + "; ".join(c["issues"]))
        return {"critiques": [dict(c, subject="plan")], "trace": [log("Critic", f"Plan validation {verdict} (attempt {c['attempt']})", detail, c)]}

    def start_practice(state: State):
        queue = [{"topic": i["topic"], "kind": i["kind"]} for i in state["plan"]["items"] if not i["deferred"]]
        return {"queue": queue, "trace": [log("Orchestrator", "Plan approved", f"practice queue: {', '.join(q['topic'] for q in queue) or 'empty'}")]}

    def next_topic(state: State):
        q = list(state.get("queue", []))
        if not q:
            return {"current": None, "trace": [log("Orchestrator", "All planned topics practised")]}
        nxt = q.pop(0)
        cur = {"topic": nxt["topic"], "kind": nxt["kind"], "round": 1, "seen": [], "missed": [], "missed_q": []}
        return {"current": cur, "queue": q, "quiz_attempts": 0,
                "trace": [log("Orchestrator", f"Next topic: {TOPICS[nxt['topic']]['name']}", f"{nxt['kind']}; {len(q)} more after this")]}

    def teach(state: State):
        cur, lid = state["current"], state["learner_id"]
        topic, rnd = cur["topic"], cur["round"]
        live = T.mastery_map(lid, state.get("history_add", []))[topic]
        first = T.get_learner(lid)["name"].split()[0]
        note = ""
        if cur["kind"] == "challenge" and rnd == 1:
            text, source = "Challenge round: no new teaching. Prove your mastery on harder questions.", "none"
        else:
            text, source, note = U.explain(helper if use_llm_explain else None, first, topic, live, rnd, cur.get("missed_q", []))
        level = "beginner" if rnd > 1 or live["mastery"] < 0.5 else "intermediate"
        resources = T.find_resources.invoke({"topic": topic, "level": level})["resources"][:2]
        cur2 = dict(cur, material={"explanation": text, "source": source, "resources": resources})
        trace = [log("Tutor", f"{'Taught' if rnd == 1 else 'Re-taught'} {TOPICS[topic]['name']} (round {rnd})",
                     f"explanation by {source}; resources: {', '.join(r['id'] for r in resources)}", cur2["material"])]
        if note:
            trace.append(log("Tutor", "Fallback", note))
        return {"current": cur2, "quiz_attempts": 0, "trace": trace}

    def make_quiz(state: State):
        cur, lid = state["current"], state["learner_id"]
        topic, attempt = cur["topic"], state.get("quiz_attempts", 0)
        live = T.mastery_map(lid, state.get("history_add", []))[topic]["mastery"]
        wanted = U.quiz_difficulties(live, cur["kind"])
        trace, qs, source = [], None, "bank"
        if helper is not None and use_llm_questions and attempt == 0:
            try:
                avoid = [q["text"] for q in QUESTIONS.values() if q["topic"] == topic]
                qs, source = U.llm_quiz(helper, topic, wanted, avoid), "llm"
            except Exception as e:
                trace.append(log("Tutor", "Fallback", f"LLM question writing failed ({type(e).__name__}); using the question bank"))
        if qs is None:
            qs = U.bank_quiz(topic, wanted, cur["seen"], cur["missed"])
        faulted = False
        if state.get("inject_fault") and not state.get("quiz_fault_done") and attempt == 0:
            qs, faulted = U.corrupt_quiz(qs), True
        quiz = {"topic": topic, "round": cur["round"], "wanted": wanted, "source": source, "questions": qs}
        detail = f"{len(qs)} questions from the {source}, difficulty {wanted}" + (" (corrupted on purpose)" if faulted else "")
        trace.append(log("Tutor", f"Prepared quiz (attempt {attempt + 1})", detail, [q["id"] for q in qs]))
        return {"quiz": quiz, "quiz_attempts": attempt + 1, "quiz_fault_done": bool(state.get("quiz_fault_done")) or faulted, "trace": trace}

    def critic_quiz(state: State):
        cur, quiz = state["current"], state["quiz"]
        gen_helper = helper if quiz["source"] == "llm" else None
        c = C.check_quiz(cur["topic"], quiz, quiz["wanted"], set(cur["seen"]), set(cur["missed"]), state["quiz_attempts"], gen_helper)
        ok = sum(x["passed"] for x in c["checks"])
        verdict = "PASSED" if c["passed"] else "FAILED"
        detail = f"{ok}/{len(c['checks'])} checks passed" + ("" if c["passed"] else ": " + "; ".join(c["issues"]))
        return {"critiques": [dict(c, subject=f"quiz {cur['topic']}")], "trace": [log("Critic", f"Quiz validation {verdict} (attempt {c['attempt']})", detail, c)]}

    def ask(state: State):
        cur, quiz = state["current"], state["quiz"]
        payload = {"topic": cur["topic"], "kind": cur["kind"], "round": cur["round"], "material": cur["material"],
                   "questions": U.public_questions(quiz["questions"])}
        answers = interrupt(payload)
        stopped = isinstance(answers, dict) and bool(answers.get("__stop__"))
        return {"answers": {} if stopped else answers, "stopped": stopped,
                "trace": [log("Orchestrator", "Learner ended the session" if stopped else "Learner answered", f"{TOPICS[cur['topic']]['name']}, round {cur['round']}")]}

    def assess(state: State):
        cur, lid = state["current"], state["learner_id"]
        res, hist = AS.assess(lid, cur, state["quiz"], state["answers"], state.get("history_add", []))
        cur2 = dict(cur, seen=cur["seen"] + [q["id"] for q in state["quiz"]["questions"]], missed=res["missed"],
                    missed_q=[q for q in state["quiz"]["questions"] if q["id"] in res["missed"]])
        if res["decision"] == "reteach":
            cur2["round"] = cur["round"] + 1
        word = {"cleared": "CLEARED", "reteach": "RE-TEACH", "escalate": "ESCALATE"}[res["decision"]]
        detail = f"score {res['correct']}/{res['total']}; mastery {round(res['mastery_before'] * 100)}% → {round(res['mastery_after'] * 100)}%"
        return {"results": state["results"] + [res], "history_add": hist, "current": cur2, "decision": res["decision"],
                "trace": [log("Assessor", f"{TOPICS[cur['topic']]['name']}: {word}", detail, res)]}

    def escalate_topic(state: State):
        cur = state["current"]
        out = T.escalate_to_tutor.invoke({"learner_id": state["learner_id"], "topic": cur["topic"],
                                           "reason": f"Stuck on {TOPICS[cur['topic']]['name']} after {AS.MAX_ROUNDS} rounds"})
        return {"tickets": state["tickets"] + [{"topic": cur["topic"], "ticket_id": out["ticket_id"]}],
                "trace": [log("Orchestrator", f"Escalated {cur['topic']} to a human tutor", f"ticket {out['ticket_id']}", out)]}

    def finalize(state: State):
        status = "stopped" if state.get("stopped") else "completed"
        return {"status": status, "final": R.build_report(state), "trace": [log("Orchestrator", f"Session {status}: report ready")]}

    def escalate(state: State):
        lid = state["learner_id"]
        failed = state.get("critiques") and not state["critiques"][-1]["passed"]
        reason = ("The Critic could not validate the study plan: " + "; ".join(state["critiques"][-1]["issues"])) if failed \
            else f"Learner asked for a human tutor: {state['request']}"
        out = T.escalate_to_tutor.invoke({"learner_id": lid, "topic": "(general)", "reason": reason})
        text = (f"I've opened ticket **{out['ticket_id']}** so a human tutor can follow up with you. "
                "I can't promise a specific outcome, but they will see your request.")
        return {"status": "escalated", "final": text, "tickets": [{"topic": "(general)", "ticket_id": out["ticket_id"]}],
                "trace": [log("Orchestrator", f"Escalated: ticket {out['ticket_id']}", reason, out)]}

    def refuse(state: State):
        text = REFUSALS.get(state["intent"]) or state["rationale"]
        if state["intent"] == "out_of_scope":
            text += "\n\nTry something like: *I want to understand backpropagation in 10 days, 1.5 hours a day.*"
        return {"status": "refused", "final": text, "trace": [log("Orchestrator", "Refused", state["rationale"])]}

    def route_goal(state: State):
        return {"privacy": "refuse", "out_of_scope": "refuse", "escalate": "escalate"}.get(state["intent"], "diagnose")

    def route_plan(state: State):
        if state["critiques"][-1]["passed"]:
            return "start"
        return "retry" if state["plan_attempts"] < MAX_PLAN_ATTEMPTS else "escalate"

    def route_topic(state: State):
        return "teach" if state.get("current") else "finalize"

    def route_quiz(state: State):
        if state["critiques"][-1]["passed"]:
            return "ask"
        return "retry" if state["quiz_attempts"] < MAX_QUIZ_ATTEMPTS else "stuck"

    def route_ask(state: State):
        return "finalize" if state.get("stopped") else "assess"

    def route_assess(state: State):
        return {"cleared": "next_topic", "reteach": "teach", "escalate": "escalate_topic"}[state["decision"]]

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("diagnose", diagnose), ("tutor_plan", tutor_plan), ("critic_plan", critic_plan),
                     ("start_practice", start_practice), ("next_topic", next_topic), ("teach", teach), ("make_quiz", make_quiz),
                     ("critic_quiz", critic_quiz), ("ask", ask), ("assess", assess), ("escalate_topic", escalate_topic),
                     ("finalize", finalize), ("escalate", escalate), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_goal, {"refuse": "refuse", "escalate": "escalate", "diagnose": "diagnose"})
    g.add_edge("diagnose", "tutor_plan")
    g.add_edge("tutor_plan", "critic_plan")
    g.add_conditional_edges("critic_plan", route_plan, {"start": "start_practice", "retry": "tutor_plan", "escalate": "escalate"})
    g.add_edge("start_practice", "next_topic")
    g.add_conditional_edges("next_topic", route_topic, {"teach": "teach", "finalize": "finalize"})
    g.add_edge("teach", "make_quiz")
    g.add_edge("make_quiz", "critic_quiz")
    g.add_conditional_edges("critic_quiz", route_quiz, {"ask": "ask", "retry": "make_quiz", "stuck": "escalate_topic"})
    g.add_conditional_edges("ask", route_ask, {"assess": "assess", "finalize": "finalize"})
    g.add_conditional_edges("assess", route_assess, {"next_topic": "next_topic", "teach": "teach", "escalate_topic": "escalate_topic"})
    g.add_edge("escalate_topic", "next_topic")
    for end_node in ("finalize", "escalate", "refuse"):
        g.add_edge(end_node, END)
    return g.compile(checkpointer=checkpointer or MemorySaver())

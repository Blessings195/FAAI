"""LearnPath AI: Streamlit UI.  Run:  streamlit run app.py"""
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from learnpath import report as R
from learnpath import session as S
from learnpath import tools as T
from learnpath import viz
from learnpath.diagram import DOT
from learnpath.evaluation import run_evaluation
from learnpath.llm import DEFAULT_OLLAMA_URL, LLMHelper, get_llm, ollama_models
from learnpath.simulator import SimLearner
from learnpath.workflow import build_workflow

load_dotenv()
st.set_page_config(page_title="LearnPath AI", page_icon="🧭", layout="wide")

ICONS = {"Planner": "🧭", "Diagnostic": "🔎", "Tutor": "🎓", "Critic": "🛡️", "Assessor": "📝", "Orchestrator": "🧩"}
TOPICS = T.TOPICS
EXAMPLES = [
    ("L001", "I want to understand backpropagation in 10 days, 1.5 hours a day"),
    ("L003", "I want to learn linear regression in 4 days, 1 hour a day"),
    ("L004", "I want to master neural networks in 3 days, 1 hour a day"),
    ("L002", "Teach me gradient descent"),
    ("L001", "Show me Rahul's quiz scores"),
    ("L003", "I'm stuck, I want to talk to a human tutor"),
]


@st.cache_resource(show_spinner=False)
def get_graph(mode, model, base_url, goal_on, explain_on, questions_on):
    llm = get_llm(mode, model, base_url)
    helper = LLMHelper(llm, f"{mode}:{model}") if llm is not None else None
    graph = build_workflow(helper, use_llm_goal=goal_on, use_llm_explain=explain_on, use_llm_questions=questions_on)
    return graph, helper


def set_example(lid, text):
    st.session_state["learner"], st.session_state["goal"] = lid, text


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🧭 LearnPath AI")
    st.caption("Personalized learning and knowledge-gap remediation · dummy data")
    st.selectbox("Learner", list(T.LEARNERS), key="learner",
                 format_func=lambda k: f"{k} · {T.LEARNERS[k]['name']} ({T.LEARNERS[k]['level']})")
    mode = st.radio("Model mode", ["mock", "ollama", "anthropic"], index=0,
                    help="mock = deterministic agents, no LLM, no key, no internet needed")
    model, base_url = "", DEFAULT_OLLAMA_URL
    goal_on = explain_on = questions_on = False
    if mode == "ollama":
        base_url = st.text_input("Ollama URL", DEFAULT_OLLAMA_URL)
        if st.button("Check Ollama connection", key="ollama_check"):
            st.session_state["ollama"] = ollama_models(base_url)
        found = st.session_state.get("ollama")
        if found and found[0] and found[1]:
            model = st.selectbox("Installed model", found[1])
            st.success(f"Connected: {len(found[1])} model(s) found")
        else:
            model = st.text_input("Model name", "llama3.1")
            if found and not found[0]:
                st.error("Ollama is not reachable. Start it and pull a model, e.g. `ollama pull llama3.1`.")
            elif found:
                st.warning("Ollama is running but has no models. Run `ollama pull llama3.1`.")
            else:
                st.caption("Press the button to list installed models.")
    elif mode == "anthropic":
        model = st.text_input("Model name", "claude-sonnet-5-5")
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("ANTHROPIC_API_KEY is not set (put it in a .env file).")
    if mode != "mock":
        st.caption("What the LLM may do (the Critic checks all of it; every use has a rule-based fallback)")
        goal_on = st.checkbox("Parse my goal", value=True)
        explain_on = st.checkbox("Write explanations", value=True)
        questions_on = st.checkbox("Write extra practice questions (slower)", value=False)
    practice = st.radio("Practice mode", ["I answer the questions", "Autopilot (simulated learner)"], index=0)
    inject = st.checkbox("Inject faulty first drafts (demo the Critic)", value=False,
                         help="The first study plan and the first quiz are deliberately wrong. The Critic should reject them.")
    st.divider()
    st.caption("Try an example")
    for i, (lid, text) in enumerate(EXAMPLES):
        st.button(text, key=f"ex{i}", on_click=set_example, args=(lid, text), width="stretch")

st.title("LearnPath AI")
st.write("A **Planner** turns your goal into tasks, a **Diagnostic** agent finds your knowledge gaps and their root causes, "
         "a **Tutor** builds your path and teaches, and a **Critic** validates every plan and quiz before you see it.")

if "goal" not in st.session_state:
    st.session_state["goal"] = T.LEARNERS["L001"]["default_goal"]
st.text_input("What do you want to learn, and how much time do you have?", key="goal")
run = st.button("Build my learning path", type="primary", key="run")

# ------------------------------------------------------------------ run
try:
    graph, helper = get_graph(mode, model, base_url, goal_on, explain_on, questions_on)
except Exception as e:
    st.warning(f"Could not start '{mode}' mode ({type(e).__name__}: {e}). Using mock mode instead.")
    graph, helper = get_graph("mock", "", DEFAULT_OLLAMA_URL, False, False, False)

if run and st.session_state.get("goal", "").strip():
    t0 = time.perf_counter()
    lid = st.session_state["learner"]
    with st.status("Agents are working...", expanded=True) as status:
        def show(node, update):
            for e in update.get("trace", []):
                st.write(f"{ICONS.get(e['agent'], '•')} **{e['agent']}**: {e['action']}")
        state, pending, tid = S.start(graph, st.session_state["goal"], lid, inject, show)
        if pending and practice.startswith("Autopilot"):
            state = S.autopilot(graph, state, pending, tid, SimLearner(lid), show)
            pending = None
        status.update(label=f"Done in {time.perf_counter() - t0:.1f}s", state="complete", expanded=False)
    st.session_state["lp"] = {"state": state, "pending": pending, "tid": tid, "graph": graph}

lp = st.session_state.get("lp")
state = lp["state"] if lp else None

tabs = st.tabs(["Learning path", "Knowledge gaps", "Practice", "Report", "Agent trace", "Critic validation",
                "Architecture", "Evaluation"])

# ------------------------------------------------------------------ learning path
with tabs[0]:
    if not state:
        st.info("Pick an example in the sidebar or type a goal, then press **Build my learning path**.")
    elif state.get("status") in ("refused", "escalated") and "plan" not in state:
        st.warning(state["final"]) if state["status"] == "refused" else st.success(state["final"])
    else:
        plan, goal, diag = state["plan"], state["goal"], state["diagnosis"]
        c = st.columns(5)
        c[0].metric("Mode", diag["mode"])
        c[1].metric("Topics to cover", len([i for i in plan["items"] if not i["deferred"]]))
        c[2].metric("Time budget", f"{plan['budget_hours']:g} h")
        c[3].metric("Time needed", f"{plan['needed_hours']:g} h")
        c[4].metric("Feasible", "Yes" if plan["feasible"] else "No")
        if not plan["feasible"]:
            st.warning(f"You need {plan['needed_hours']:g} h but have {plan['budget_hours']:g} h. Extend to "
                       f"**{plan['days_needed']} days** to cover everything; until then the plan fixes root causes first.")
        for a in goal["assumptions"]:
            st.caption(f"Assumption: {a}")
        st.subheader("Day-by-day path")
        rows = R.day_table(plan)
        st.dataframe(pd.DataFrame(rows, columns=["Day", "Topic", "Activity", "Resource", "Minutes"]), hide_index=True, width="stretch")
        st.subheader("Why these topics")
        st.dataframe(pd.DataFrame([{"Topic": TOPICS[i["topic"]]["name"], "Kind": i["kind"], "Role": i["role"],
                                    "Mastery": f"{round(i['mastery'] * 100)}%", "Hours needed": i["hours_needed"],
                                    "Hours planned": i["minutes"] / 60, "Note": "deferred" if i["deferred"] else "partial" if i["partial"] else ""}
                                   for i in plan["items"]]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ knowledge gaps
with tabs[1]:
    if not state or "diagnosis" not in state:
        st.info("Run a learning goal to see the knowledge-gap analysis.")
    else:
        diag = state["diagnosis"]
        live = T.mastery_map(state["learner_id"], state.get("history_add", []))
        left, right = st.columns(2)
        left.subheader("At the start")
        left.graphviz_chart(viz.knowledge_graph_dot(state["mastery_base"], state["goal"]["targets"]), width="stretch")
        right.subheader("Now")
        right.graphviz_chart(viz.knowledge_graph_dot(live, state["goal"]["targets"]), width="stretch")
        st.caption("Green = mastered, amber = shaky, red = gap, grey = not yet assessed. Thick teal border = your goal topics.")
        if diag["mode"] == "remediate":
            st.markdown(f"**Root causes:** {', '.join(f'{t} ({TOPICS[t]['name']})' for t in diag['roots'])}. "
                        + ("Downstream gaps: " + "; ".join(diag["chains"]) + "." if diag["chains"] else ""))
        else:
            st.markdown("**No gaps** in the goal topics: challenge mode with a stretch topic.")
        rows = R.progress_rows(state)
        df = pd.DataFrame([{"Topic": r["name"], "Role": r["role"], "Start": r["before"], "Now": r["after"], "Status": r["status"]} for r in rows])
        st.bar_chart(df.set_index("Topic")[["Start", "Now"]], horizontal=True)
        st.dataframe(df.assign(Start=df["Start"].map(R.pct), Now=df["Now"].map(R.pct)), hide_index=True, width="stretch")

# ------------------------------------------------------------------ practice
with tabs[2]:
    if not state or "plan" not in state:
        st.info("Build a learning path first, then practise here.")
    else:
        pend = lp["pending"]
        if pend:
            topic = TOPICS[pend["topic"]]["name"]
            st.subheader(f"{topic} · round {pend['round']}")
            if pend["round"] > 1:
                st.info("Re-teach round: new explanation, easier resources, and a question you missed before.")
            with st.expander("Study material", expanded=True):
                st.markdown(pend["material"]["explanation"])
                st.caption(f"Explanation by: {pend['material']['source']}")
                for r in pend["material"]["resources"]:
                    st.markdown(f"- {r['type'].title()}: {r['title']} ({r['minutes']} min, {r['level']})")
            key = f"{lp['tid']}-{len(state.get('results', []))}"
            with st.form(key=f"quiz-{key}"):
                for n, q in enumerate(pend["questions"], 1):
                    st.radio(f"Q{n}. {q['text']}", list(range(len(q["options"]))), index=None,
                             format_func=lambda i, o=q["options"]: o[i], key=f"ans-{key}-{q['id']}")
                submitted = st.form_submit_button("Submit answers", type="primary")
            if st.button("End session and show my report", key=f"stop-{key}"):
                with st.spinner("Wrapping up..."):
                    state, pending = S.resume(lp["graph"], state, lp["tid"], {"__stop__": True})
                lp.update(state=state, pending=pending)
                st.rerun()
            if submitted:
                answers = {q["id"]: st.session_state.get(f"ans-{key}-{q['id']}") for q in pend["questions"]}
                with st.spinner("Grading and planning the next step..."):
                    state, pending = S.resume(lp["graph"], state, lp["tid"], answers)
                lp.update(state=state, pending=pending)
                st.rerun()
        elif state.get("status") in ("completed", "stopped"):
            st.success("Session finished. See the **Report** tab for your results and next steps.")
        results = state.get("results", [])
        if results:
            st.subheader("Latest result" if pend else "Results")
            last = results[-1]
            word = {"cleared": "✅ Cleared", "reteach": "🔁 Needs another round", "escalate": "🎫 Sent to a human tutor"}[last["decision"]]
            st.markdown(f"**{TOPICS[last['topic']]['name']}**, round {last['round']}: {last['correct']}/{last['total']} · {word} · "
                        f"mastery {R.pct(last['mastery_before'])} → {R.pct(last['mastery_after'])}")
            for f in last["feedback"]:
                st.markdown(f"{'✅' if f['correct'] else '❌'} {f['text']}  \n_Correct answer: {f['correct_option']}. {f['explanation']}_")
            with st.expander("All rounds so far"):
                st.dataframe(pd.DataFrame([{"Topic": TOPICS[r["topic"]]["name"], "Round": r["round"], "Score": f"{r['correct']}/{r['total']}",
                                            "Decision": r["decision"], "Mastery": f"{R.pct(r['mastery_before'])} → {R.pct(r['mastery_after'])}"}
                                           for r in results]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ report
with tabs[3]:
    if not state:
        st.info("Your report appears here.")
    elif state.get("final") and state.get("status") in ("completed", "stopped", "refused", "escalated"):
        st.markdown(state["final"])
    else:
        st.info("Finish the practice session (or press *End session*) to get your report.")

# ------------------------------------------------------------------ trace
with tabs[4]:
    if not state:
        st.info("Run a goal to see every agent step.")
    else:
        if helper is not None:
            st.caption(f"LLM ({helper.label}): {helper.calls} call(s), {helper.failures} failure(s). Failures fall back to rules.")
        else:
            st.caption("Mock mode: no LLM calls.")
        if state.get("tasks"):
            st.subheader("Task plan")
            st.dataframe(pd.DataFrame(state["tasks"])[["step", "agent", "tool", "purpose"]], hide_index=True, width="stretch")
        st.subheader("Execution trace")
        for e in state["trace"]:
            st.markdown(f"{ICONS.get(e['agent'], '•')} **{e['agent']}** · {e['action']}")
            if e["detail"]:
                st.caption(e["detail"])
            if e["data"] is not None:
                with st.expander("details"):
                    st.json(e["data"], expanded=False)

# ------------------------------------------------------------------ critic
with tabs[5]:
    if not state or not state.get("critiques"):
        st.info("No Critic output yet (the request was refused or escalated before validation).")
    else:
        st.write(f"The Critic ran {len(state['critiques'])} validation round(s); "
                 f"{sum(1 for c in state['critiques'] if not c['passed'])} were rejected.")
        for c in state["critiques"]:
            st.markdown(f"#### {c['subject'].title()} · attempt {c['attempt']}: {'✅ PASSED' if c['passed'] else '❌ FAILED'}")
            df = pd.DataFrame(c["checks"])
            df["passed"] = df["passed"].map({True: "✅", False: "❌"})
            st.dataframe(df.rename(columns={"check": "Check", "passed": "Result", "detail": "Detail"}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ architecture
with tabs[6]:
    st.graphviz_chart(DOT, width="stretch")
    st.markdown(pd.DataFrame([
        ("Planner", "Parses the goal (topics, days, hours per day), routes: refuse, escalate or learn, writes the task plan"),
        ("Diagnostic", "Builds the mastery map, walks the prerequisite graph, separates root-cause gaps from downstream gaps"),
        ("Tutor", "Builds the time-boxed study path, explains topics, prepares quizzes, re-teaches when a quiz is failed"),
        ("Critic / Assessor", "Re-derives the facts to validate the plan and each quiz, grades answers, decides cleared / re-teach / escalate"),
        ("Orchestrator", "LangGraph state machine with a human-in-the-loop pause for the learner's answers, retry loops and escalation"),
    ], columns=["Agent", "Responsibility"]).to_markdown(index=False))
    st.subheader("Optional LLM (Ollama or API)")
    st.markdown(pd.DataFrame([
        ("Planner", "Parses a free-text goal into topics and a time budget", "Rule-based parser"),
        ("Tutor", "Writes the explanation for each topic and re-teach round", "Template from the topic notes"),
        ("Tutor", "Writes extra practice questions (off by default)", "Curated question bank"),
        ("Critic", "Answers LLM-written questions independently to verify the answer key", "Question is rejected"),
    ], columns=["Agent", "What the LLM does", "If it fails or is off"]).to_markdown(index=False))
    with st.expander("Dummy data"):
        st.dataframe(pd.DataFrame([{"ID": k, "Name": v["name"], "Level": v["level"], "Quiz attempts": len(v["history"]),
                                    "Default goal": v["default_goal"]} for k, v in T.LEARNERS.items()]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{"ID": t["id"], "Topic": t["name"], "Prerequisites": ", ".join(t["prerequisites"]) or "-",
                                    "Est. hours": t["est_hours"]} for t in TOPICS.values()]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ evaluation
with tabs[7]:
    st.write("Runs 11 scenario tests with a simulated learner and a fault-injection test of the Critic, in mock mode.")
    if st.button("Run evaluation suite", key="eval_btn"):
        with st.spinner("Running..."):
            st.session_state["eval_results"] = run_evaluation()
    if "eval_results" in st.session_state:
        scen, fault = st.session_state["eval_results"]
        m = st.columns(2)
        m[0].metric("Scenario tests passed", f"{(scen['Result'] == 'PASS').sum()}/{len(scen)}")
        m[1].metric("Bad drafts caught and recovered", f"{(fault['Result'] == 'PASS').sum()}/{len(fault)}")
        st.dataframe(scen, hide_index=True, width="stretch")
        st.caption("Fault injection: the first study plan and the first quiz are corrupted; the Critic must reject them and the outcome must match a clean run.")
        st.dataframe(fault, hide_index=True, width="stretch")

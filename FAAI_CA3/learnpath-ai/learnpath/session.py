"""Drivers for running the graph: start, resume after the learner answers, and an autopilot with a simulated learner."""
import uuid

from langgraph.types import Command


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 400}


def _merge(state, update):
    for k, v in update.items():
        if k in ("trace", "critiques"):
            state[k] = state.get(k, []) + v
        else:
            state[k] = v


def _consume(stream, state, on_update):
    pending = None
    for chunk in stream:
        if "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
            continue
        for node, update in chunk.items():
            if update:
                _merge(state, update)
                if on_update:
                    on_update(node, update)
    return pending


def start(graph, request, learner_id, inject_fault=False, on_update=None):
    """Run until the first question is ready (pending is not None) or the run ends. Returns (state, pending, thread_id)."""
    tid = uuid.uuid4().hex
    state = {"request": request, "learner_id": learner_id, "inject_fault": inject_fault, "trace": [], "critiques": []}
    pending = _consume(graph.stream(dict(state), _config(tid), stream_mode="updates"), state, on_update)
    return state, pending, tid


def resume(graph, state, thread_id, answers, on_update=None):
    """Give the learner's answers (or {"__stop__": True}) to the waiting graph. Returns (state, pending)."""
    pending = _consume(graph.stream(Command(resume=answers), _config(thread_id), stream_mode="updates"), state, on_update)
    return state, pending


def autopilot(graph, state, pending, thread_id, sim, on_update=None, max_rounds=200):
    """Answer every question with a simulated learner until the session ends."""
    n = 0
    while pending and n < max_rounds:
        answers = sim.answer(state["quiz"], pending["topic"], pending["round"], pending["kind"])
        state, pending = resume(graph, state, thread_id, answers, on_update)
        n += 1
    return state

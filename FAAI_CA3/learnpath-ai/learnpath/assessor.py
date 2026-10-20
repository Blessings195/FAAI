"""Assessor (part of the Critic Agent): grades answers, updates mastery and decides: cleared, re-teach or escalate."""
from . import tools as T

MAX_ROUNDS = 2          # teach + quiz rounds per topic before a human tutor is asked
PASS_RATIO = 2 / 3      # share of correct answers needed to clear a topic


def grade(quiz, answers):
    feedback, correct = [], 0
    for q in quiz["questions"]:
        chosen = answers.get(q["id"])
        try:
            chosen = int(chosen)
        except (TypeError, ValueError):
            chosen = None
        ok = chosen == q["answer"]
        correct += ok
        feedback.append({"id": q["id"], "text": q["text"], "chosen": chosen, "correct": ok,
                         "correct_option": q["options"][q["answer"]], "explanation": q["explanation"],
                         "difficulty": q["difficulty"]})
    return correct, feedback


def assess(learner_id, current, quiz, answers, history_add):
    """Returns (result, new_history_add). Mastery is recomputed from the stored history plus session attempts."""
    topic = current["topic"]
    correct, feedback = grade(quiz, answers)
    total = len(quiz["questions"])
    before = T.mastery_map(learner_id, history_add)[topic]["mastery"]
    new_hist = list(history_add) + [{"topic": topic, "score": correct, "max": total}]
    after = T.mastery_map(learner_id, new_hist)[topic]["mastery"]
    cleared = total > 0 and correct / total >= PASS_RATIO - 1e-9 and after >= T.MASTERED - 1e-9
    decision = "cleared" if cleared else "reteach" if current["round"] < MAX_ROUNDS else "escalate"
    result = {"topic": topic, "round": current["round"], "kind": current["kind"], "correct": correct, "total": total,
              "cleared": cleared, "decision": decision, "mastery_before": before, "mastery_after": after,
              "missed": [f["id"] for f in feedback if not f["correct"]], "feedback": feedback}
    return result, new_hist

"""Simulated learner for autopilot demos and evaluation (deterministic, no randomness)."""
from . import tools as T

THRESHOLDS = {1: 0.35, 2: 0.55, 3: 0.80}  # knowledge needed to answer a question of each difficulty


class SimLearner:
    """Knows a topic to the extent of its starting mastery plus what each teaching round adds."""

    def __init__(self, learner_id):
        spec = T.get_learner(learner_id).get("sim", {})
        self.default = spec.get("default_gain", [0.3, 0.2])
        self.overrides = spec.get("overrides", {})
        self.base = {t: v["mastery"] for t, v in T.mastery_map(learner_id).items()}

    def knowledge(self, topic, round_no, kind):
        gains = self.overrides.get(topic, self.default)
        steps = round_no - 1 if kind == "challenge" else round_no
        return self.base[topic] + sum(gains[:max(0, steps)])

    def answer(self, quiz, topic, round_no, kind):
        k = self.knowledge(topic, round_no, kind)
        answers = {}
        for q in quiz["questions"]:
            knows = k + 1e-9 >= THRESHOLDS.get(q["difficulty"], 0.8)
            key = q["answer"]
            answers[q["id"]] = key if (knows and isinstance(key, int)) else next(
                i for i in range(len(q["options"])) if i != key)
        return answers

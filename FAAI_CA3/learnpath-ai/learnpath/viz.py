"""Visual helpers: prerequisite graph coloured by mastery (Graphviz DOT) and before/after tables."""
from .tools import TOPICS, TOPIC_ORDER

COLORS = {"mastered": "#c8e6c9", "shaky": "#fff3cd", "gap": "#f8c9c4", "unassessed": "#e0e0e0"}


def knowledge_graph_dot(mastery, targets=(), highlight=()):
    lines = ['digraph G { rankdir=LR; nodesep=0.25; ranksep=0.5;',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#555555"];',
             'edge [color="#777777"];']
    for t in TOPIC_ORDER:
        m = mastery[t]
        label = f"{TOPICS[t]['name']}\\n{round(m['mastery'] * 100)}% ({m['status']})"
        extra = ', penwidth=3, color="#0F6E6E"' if t in targets else ""
        if t in highlight:
            extra += ', peripheries=2'
        lines.append(f'{t} [label="{label}", fillcolor="{COLORS[m["status"]]}"{extra}];')
    for t in TOPIC_ORDER:
        for p in TOPICS[t]["prerequisites"]:
            lines.append(f"{p} -> {t};")
    lines.append("}")
    return "\n".join(lines)

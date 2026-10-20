"""Architecture diagram (Graphviz DOT, rendered by Streamlit and used in the report and slides)."""

DOT = r"""
digraph G {
  rankdir=TB; nodesep=0.15; ranksep=0.3; pad=0.05;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, color="#555555", margin="0.1,0.05"];
  edge [fontname="Helvetica", fontsize=13, color="#555555"];
  ui [label="Streamlit UI\nlearner + goal", fillcolor="#e8f0fe"];
  llm [label="LLM (optional)\nOllama / API / mock", shape=component, fillcolor="#fffde7"];
  planner [label="Planner Agent\ngoal + task plan", fillcolor="#fff3cd"];
  d1 [label="route?", shape=diamond, fillcolor="#ffffff"];
  refuse [label="Refuse", fillcolor="#eeeeee"];
  diag [label="Diagnostic Agent\nmastery + root causes", fillcolor="#d9f2e6"];
  tools [label="Tools\nprofile, mastery, graph,\nresources, questions", style="rounded,filled,dashed", fillcolor="#f5f5f5"];
  data [label="Dummy data\nlearners, topics, questions", shape=cylinder, fillcolor="#f5f5f5"];
  tutor [label="Tutor Agent\nplan, teach, quiz", fillcolor="#fde2e4"];
  critic [label="Critic Agent\nvalidates plan + quiz", fillcolor="#e2d9f3"];
  d2 [label="valid?", shape=diamond, fillcolor="#ffffff"];
  learner [label="Learner answers\n(or simulated learner)", fillcolor="#e8f0fe"];
  assess [label="Assessor\ngrade + update mastery", fillcolor="#e2d9f3"];
  d3 [label="cleared?", shape=diamond, fillcolor="#ffffff"];
  human [label="Escalate\nhuman tutor ticket", fillcolor="#ffe0b2"];
  report [label="Progress report", fillcolor="#c8e6c9"];

  ui -> planner [label=" goal"];
  llm -> planner [style=dotted, arrowhead=none];
  llm -> tutor [style=dotted, arrowhead=none];
  planner -> d1;
  d1 -> refuse [label=" refuse"];
  d1 -> human [label=" human"];
  d1 -> diag [label=" learn"];
  diag -> tools [dir=both];
  tools -> data;
  diag -> tutor [label=" gaps"];
  tutor -> critic [label=" plan, quiz"];
  critic -> tools [style=dashed, label=" re-verify"];
  critic -> d2;
  d2 -> tutor [label=" fail", color="#c62828", fontcolor="#c62828"];
  d2 -> learner [label=" pass"];
  learner -> assess [label=" answers"];
  assess -> d3;
  d3 -> tutor [label=" no: re-teach", color="#c62828", fontcolor="#c62828"];
  d3 -> human [label=" stuck"];
  d3 -> report [label=" yes"];
  {rank=same; planner; llm}
}
"""

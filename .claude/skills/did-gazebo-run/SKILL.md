---
name: did-gazebo-run
description: Run one DID Hack mission in Gazebo headless (WSL, ROS 2 Jazzy) with a time limit and summarise the agent's experiment journal, LLM decisions and score. Use to prove a change works in the real simulation, not only offline.
---

# Gazebo mission with proof

The assistant cannot see the Gazebo window; the proof is the log. This skill turns "run it and look" into one call.

1. Make sure nothing else is running: `pgrep -af "gz sim|ros2 launch"`. If the user's own simulation is running, ask before stopping it.
   Never use `pkill -f <pattern>` from a `bash -lc` string — it matches its own shell. Kill by PID.
2. Build if Python or launch files changed: `source /opt/ros/jazzy/setup.bash && colcon build --symlink-install`.
3. Run with a time limit (seconds) and launch arguments:
   ```bash
   ./scripts/gz_run.sh 420 scenario:=hard planner:=science
   ./scripts/gz_run.sh 480 scenario:=hard planner:=science llm:=true battery:=30
   ./scripts/gz_run.sh 420 scenario:=medium planner:=science knowledge:=runs/knowledge.json
   ```
   The script stops the simulation itself (`scripts/stop_sim.sh`) and prints: hypotheses / conclusions / adaptations /
   decisions / LLM choices from the journal, the final score JSON and the number of tracebacks.
4. Report to the user: samples collected / total, returned or not, penalties, score, time, and 2–3 notable journal lines
   (a hypothesis confirmed, an environment change detected, an LLM decision). A traceback count > 0 must be investigated
   in `/tmp/did_gz_run.log` (`grep -A30 Traceback`), except the known harmless one on shutdown.
5. Run artefacts: `runs/<time>_agent.json`, `runs/<time>_lab.md` (experiment journal), `runs/<time>_judge_*.json`,
   `runs/knowledge.json` (knowledge base for the next mission).

Tell the user what to check visually if they watch RViz or the control panel (orange zones = learned terrain,
violet = unexplored area, ring with "?" = current sample estimate).

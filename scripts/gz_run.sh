#!/usr/bin/env bash
# Headless Gazebo mission with a time limit, then a short summary (journal, LLM decisions, score).
#   ./scripts/gz_run.sh 420 scenario:=hard planner:=science [llm:=true] [battery:=30]
# Log: /tmp/did_gz_run.log. Stops everything at the end (scripts/stop_sim.sh).
DUR=${1:-420}; shift
cd "$(dirname "$0")/.." || exit 1
source /opt/ros/jazzy/setup.bash && source install/setup.bash
LOG=/tmp/did_gz_run.log
timeout --signal=INT "$DUR" ros2 launch did_agent did.launch.py rviz:=false "$@" > "$LOG" 2>&1
sleep 3
./scripts/stop_sim.sh > /dev/null 2>&1
echo "== $* (log $LOG)"
grep -E "журнал .*(гипотеза|вывод|адаптация|решение|LLM|знания)" "$LOG" | sed 's/.*\[did_agent\]: //' | cut -c1-200
grep -E "mission done" "$LOG" | sed 's/.*mission done. score: //'
grep -c "Traceback" "$LOG" | sed 's/^/tracebacks: /'

#!/usr/bin/env bash
# Web control panel on http://localhost:8080 (open it in the Windows browser).
#   ./scripts/control_panel.sh            # port 8080
#   ./scripts/control_panel.sh 8090       # another port
cd "$(dirname "$0")/.." || exit 1
source /opt/ros/jazzy/setup.bash
source install/setup.bash
exec ros2 run did_agent control_panel --port "${1:-8080}" --root "$PWD"

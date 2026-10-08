#!/usr/bin/env bash
# Stop everything started by did.launch.py / sim.launch.py / turtlebot3_world.launch.py.
pkill -INT -f "ros2 launch (did_agent|turtlebot3_gazebo)"
for i in $(seq 1 10); do
  pgrep -f "gz sim -r -s" >/dev/null || break
  sleep 1
done
# anything left over (e.g. after a crash)
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "lib/did_agent/(agent|judge)" 2>/dev/null
pkill -9 -x rviz2 2>/dev/null
pkill -9 -f parameter_bridge 2>/dev/null
pkill -9 -f "robot_state_publisher|static_transform_publisher|map_server|lifecycle_manager" 2>/dev/null
sleep 1
if pgrep -f "gz sim" >/dev/null; then echo "still running:"; pgrep -af "gz sim"; exit 1; fi
echo "simulation stopped"

#!/usr/bin/env bash
# Same as RViz "2D Goal Pose": drive to (x, y) now (world/map frame). Goal at the base (-2.0 -0.5) = return and finish.
#   ./scripts/send_goal.sh 0.55 -0.55
[ $# -eq 2 ] || { echo "usage: $0 X Y"; exit 1; }
source /opt/ros/jazzy/setup.bash
ros2 topic pub --once -w 1 /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: $1, y: $2}, orientation: {w: 1.0}}}" >/dev/null && echo "goal ($1, $2) sent"

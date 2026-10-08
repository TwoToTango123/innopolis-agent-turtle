#!/usr/bin/env bash
# Same as clicking RViz "Publish Point" several times: append points to the route.
#   ./scripts/send_route.sh 0.55,0.55 -0.55,1.6 -1.6,0.55
[ $# -ge 1 ] || { echo "usage: $0 X,Y [X,Y ...]"; exit 1; }
source /opt/ros/jazzy/setup.bash
for p in "$@"; do
  x=${p%,*}; y=${p#*,}
  ros2 topic pub --once -w 1 /clicked_point geometry_msgs/msg/PointStamped \
    "{header: {frame_id: map}, point: {x: $x, y: $y}}" >/dev/null && echo "route point ($x, $y) sent"
done

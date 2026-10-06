#!/usr/bin/env bash
# Level 0 smoke check. Run while turtlebot3_world is up:
#   TURTLEBOT3_MODEL=burger ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
# Checks topic rates, that /odom starts near zero, and odom->world vs Gazebo ground truth.
# (no set -u: ROS setup.bash uses unset variables)
source /opt/ros/jazzy/setup.bash
cd /tmp

echo "== topics"
ros2 topic list -t | grep -E "^/(odom|scan|clock|cmd_vel) " || { echo "FAIL: world not running?"; exit 1; }

for t in /odom /scan /clock; do
  printf '%-7s ' "$t"
  timeout 6 ros2 topic hz "$t" --window 20 2>/dev/null | grep -m1 "average rate" || echo "FAIL: no data"
done

python3 - <<'EOF'
import math, subprocess, re
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

rclpy.init()
node = Node('check_level0', parameter_overrides=[])
msg = []
node.create_subscription(Odometry, '/odom', lambda m: msg.append(m), 10)
while not msg:
    rclpy.spin_once(node, timeout_sec=1.0)
p = msg[-1].pose.pose
q = p.orientation
yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
wx, wy = -2.0 + p.position.x, -0.5 + p.position.y
print(f'odom          x={p.position.x:+.3f} y={p.position.y:+.3f} yaw={yaw:+.3f}')
print(f'odom->world   x={wx:+.3f} y={wy:+.3f}')
out = subprocess.run(['gz', 'model', '-m', 'burger', '-p'], capture_output=True, text=True).stdout
nums = re.findall(r'\[([-\d.]+) ([-\d.]+) ([-\d.]+)\]', out)
if nums:
    gx, gy = float(nums[0][0]), float(nums[0][1])
    gyaw = float(nums[1][2])
    err = math.hypot(gx - wx, gy - wy)
    print(f'gazebo world  x={gx:+.3f} y={gy:+.3f} yaw={gyaw:+.3f}')
    print(f'position error {err:.3f} m -> {"OK" if err < 0.05 else "CHECK"}')
node.destroy_node()
rclpy.shutdown()
EOF

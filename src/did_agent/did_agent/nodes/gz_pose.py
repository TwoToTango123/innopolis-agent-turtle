"""Ground-truth model pose straight from Gazebo transport (not via ROS).

ros_gz_bridge drops entity names when converting Pose_V, so the judge reads
/world/<world>/dynamic_pose/info with the gz-transport Python bindings.
Used ONLY by the judge; the agent must not see it.
"""
import math
import threading


class GzModelPose:
    def __init__(self, model: str = 'burger', world: str = 'default'):
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import Node
        self.model = model
        self._lock = threading.Lock()
        self._pose = None
        self._node = Node()   # keep a reference: the subscription lives as long as the node
        topic = f'/world/{world}/dynamic_pose/info'
        if not self._node.subscribe(Pose_V, topic, self._cb):
            raise RuntimeError(f'cannot subscribe to gz topic {topic}')

    def _cb(self, msg) -> None:
        for p in msg.pose:
            if p.name == self.model:
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                with self._lock:
                    self._pose = (p.position.x, p.position.y, yaw)
                return

    def get(self):
        """Latest (x, y, yaw) in the world frame, or None before the first message."""
        with self._lock:
            return self._pose

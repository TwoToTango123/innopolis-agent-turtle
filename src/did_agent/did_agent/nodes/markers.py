"""Small helpers to build RViz markers (frame: map = Gazebo world)."""
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

FRAME = 'map'


def _base(ns: str, mid: int, mtype: int, stamp) -> Marker:
    m = Marker()
    m.header.frame_id = FRAME
    m.header.stamp = stamp
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def _color(m: Marker, rgba):
    # float() matters: an int here aborts the whole process inside rosidl's C conversion
    m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
    return m


def disc(ns, mid, x, y, r, rgba, stamp, height=0.01, z=0.0) -> Marker:
    m = _base(ns, mid, Marker.CYLINDER, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z + height / 2
    m.scale.x = m.scale.y = 2.0 * r
    m.scale.z = height
    return _color(m, rgba)


def sphere(ns, mid, x, y, d, rgba, stamp, z=0.05) -> Marker:
    m = _base(ns, mid, Marker.SPHERE, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z
    m.scale.x = m.scale.y = m.scale.z = d
    return _color(m, rgba)


def text(ns, mid, x, y, s, rgba, stamp, size=0.12, z=0.25) -> Marker:
    m = _base(ns, mid, Marker.TEXT_VIEW_FACING, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z
    m.scale.z = size
    m.text = s
    return _color(m, rgba)


def arrow(ns, mid, x, y, yaw, rgba, stamp, length=0.3) -> Marker:
    import math
    m = _base(ns, mid, Marker.ARROW, stamp)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.15
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    m.scale.x, m.scale.y, m.scale.z = length, 0.05, 0.05
    return _color(m, rgba)


def cubes(ns, mid, pts, size, rgba, stamp, z=0.0) -> Marker:
    m = _base(ns, mid, Marker.CUBE_LIST, stamp)
    m.scale.x = m.scale.y = size
    m.scale.z = 0.02
    m.points = [Point(x=float(x), y=float(y), z=z) for x, y in pts]
    return _color(m, rgba)


def line(ns, mid, pts, width, rgba, stamp, z=0.03) -> Marker:
    m = _base(ns, mid, Marker.LINE_STRIP, stamp)
    m.scale.x = width
    m.points = [Point(x=float(x), y=float(y), z=z) for x, y in pts]
    return _color(m, rgba)


def delete_all(stamp) -> Marker:
    m = _base('', 0, Marker.CUBE, stamp)
    m.action = Marker.DELETEALL
    return m

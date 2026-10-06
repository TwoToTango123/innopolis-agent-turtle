"""Architecture guard: did_agent.core must stay importable without ROS."""
import os
import pkgutil
import subprocess
import sys

import did_agent.core as core

ROS_MODULES = ('rclpy', 'rosidl', 'std_msgs', 'geometry_msgs', 'sensor_msgs', 'nav_msgs', 'std_srvs')


def test_core_imports_no_ros():
    mods = [f'did_agent.core.{m.name}' for m in pkgutil.iter_modules(core.__path__)]
    assert mods, 'no core modules found'
    code = (
        'import sys\n'
        + ''.join(f'import {m}\n' for m in mods)
        + f'bad = [m for m in sys.modules if m.split(".")[0] in {ROS_MODULES!r}]\n'
        + 'print(bad); sys.exit(1 if bad else 0)\n'
    )
    # .../did_agent(pkg root)/did_agent/core/__init__.py -> pkg root
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(core.__file__))))
    r = subprocess.run([sys.executable, '-c', code], cwd=pkg_root, capture_output=True, text=True,
                       env={**os.environ, 'PYTHONPATH': pkg_root})
    assert r.returncode == 0, r.stdout + r.stderr

"""turtlebot3_world exactly as turtlebot3_gazebo/turtlebot3_world.launch.py starts it
(same world file, robot, spawn pose, bridge), but with the Gazebo GUI optional
and RViz (map + robot + scan) for watching.

  ros2 launch did_agent sim.launch.py              # headless Gazebo + RViz
  ros2 launch did_agent sim.launch.py gui:=true    # also the Gazebo window

The Gazebo GUI (Ogre2) renders garbage on Intel Arc under WSLg, see DEVLOG.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# turtlebot3 launch files read this from os.environ while being loaded
os.environ.setdefault('TURTLEBOT3_MODEL', 'burger')

START_X, START_Y = '-2.0', '-0.5'   # TASK.md: spawn point in world (= map) frame


def generate_launch_description():
    tb3_launch = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg = get_package_share_directory('did_agent')
    world = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'worlds', 'turtlebot3_world.world')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')
    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false', description='Start the Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
                                  os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            # closing the GUI must not kill the simulation
            launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'false'}.items(),
            condition=IfCondition(gui)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch, 'spawn_turtlebot3.launch.py')),
            launch_arguments={'x_pose': START_X, 'y_pose': START_Y}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tb3_launch, 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),

        # world/map frame: odom starts at the spawn point with zero yaw
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom',
             arguments=['--x', START_X, '--y', START_Y, '--frame-id', 'map', '--child-frame-id', 'odom'],
             parameters=[sim_time]),
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             parameters=[sim_time, {'yaml_filename': os.path.join(pkg, 'maps', 'map.yaml')}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_map',
             parameters=[sim_time, {'autostart': True, 'node_names': ['map_server']}]),

        Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'did.rviz')],
             parameters=[sim_time], condition=IfCondition(rviz)),
    ])

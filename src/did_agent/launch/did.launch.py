"""One command for the whole run:

  ros2 launch did_agent did.launch.py scenario:=easy
  ros2 launch did_agent did.launch.py scenario:=hard seed:=7          # generated from seed
  ros2 launch did_agent did.launch.py scenario:=/path/to/my.yaml gui:=true rviz:=false

Judge logs go to ./runs (the directory you launch from).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('did_agent')
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='easy', description='easy | medium | hard | path/to/scenario.yaml'),
        DeclareLaunchArgument('seed', default_value='-1', description='>= 0: generate the scenario from this seed'),
        DeclareLaunchArgument('gui', default_value='false', description='Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='RViz'),
        DeclareLaunchArgument('log_dir', default_value=os.path.join(os.getcwd(), 'runs')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
            launch_arguments={'gui': arg('gui'), 'rviz': arg('rviz')}.items()),

        Node(package='did_agent', executable='judge', name='did_judge', output='screen',
             parameters=[{'use_sim_time': True,
                          'scenario': arg('scenario'),
                          'seed': arg('seed'),
                          'log_dir': arg('log_dir')}]),
    ])

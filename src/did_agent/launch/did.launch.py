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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('did_agent')
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='easy', description='easy | medium | hard | path/to/scenario.yaml'),
        DeclareLaunchArgument('seed', default_value='-1', description='>= 0: generate the scenario from this seed'),
        DeclareLaunchArgument('agent', default_value='true', description='Start the agent'),
        DeclareLaunchArgument('planner', default_value='scripted', description='science: search by the sensor + terrain learning (levels 3-4) | scripted: fixed order | llm: LLM chooses targets and order | manual: goals/route from RViz'),
        DeclareLaunchArgument('learn_terrain', default_value='true', description='science: learn terrain cost from the battery (false = H1 baseline)'),
        DeclareLaunchArgument('llm_model', default_value='deepseek-v4.1-flash', description='model on ai.mai.ru (key in .env)'),
        DeclareLaunchArgument('mission', default_value='', description='mission text for the LLM planner (empty: default)'),
        DeclareLaunchArgument('battery', default_value='0.0', description='> 0: starting battery instead of 60 (tight-budget demo)'),
        DeclareLaunchArgument('gui', default_value='false', description='Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='RViz'),
        DeclareLaunchArgument('log_dir', default_value=os.path.join(os.getcwd(), 'runs')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
            launch_arguments={'gui': arg('gui'), 'rviz': arg('rviz'),
                              # the agent publishes map->odom itself
                              'static_map_odom': PythonExpression(["'false' if '", arg('agent'), "' == 'true' else 'true'"]),
                              }.items()),

        Node(package='did_agent', executable='judge', name='did_judge', output='screen',
             parameters=[{'use_sim_time': True,
                          'scenario': arg('scenario'),
                          'seed': arg('seed'),
                          'battery': ParameterValue(arg('battery'), value_type=float),
                          'log_dir': arg('log_dir')}]),

        Node(package='did_agent', executable='agent', name='did_agent', output='screen',
             condition=IfCondition(arg('agent')),
             parameters=[{'use_sim_time': True,
                          'planner': arg('planner'),
                          'learn_terrain': ParameterValue(arg('learn_terrain'), value_type=bool),
                          # level-1 demo: drive to the scenario's sample coordinates
                          'targets_from_scenario': arg('scenario'),
                          'llm_model': arg('llm_model'),
                          'mission': arg('mission'),
                          'llm_env_file': os.path.join(os.getcwd(), '.env'),
                          'seed': arg('seed'),
                          'log_dir': arg('log_dir')}]),
    ])

from glob import glob

from setuptools import find_packages, setup

package_name = 'did_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/scenarios', glob('scenarios/*.yaml')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/webui', glob('webui/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kostya',
    maintainer_email='kostya2007yaros@gmail.com',
    description='DID Hack: autonomous explorer agent and judge for TurtleBot3 in Gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'judge = did_agent.nodes.judge_node:main',
            'agent = did_agent.nodes.agent_node:main',
            'control_panel = did_agent.webui.server:main',
            'gen_scenario = did_agent.gen_scenario:main',
        ],
    },
)

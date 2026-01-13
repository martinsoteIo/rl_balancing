'''
Launch file for visualizing the robot URDF in RViz.
'''
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    ''' Launch file for visualizing the robot URDF in RViz. '''

    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            'description_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('juggling_platform_2d_description'), 'urdf', 'robot.urdf.xacro']
            ),
            description='Absolute path to robot URDF/XACRO file',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'rviz_config_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('juggling_platform_2d_description'), 'rviz', 'view_robot.rviz']
            ),
            description='RViz config file to visualize robot',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'rviz_stylesheet',
            default_value=PathJoinSubstitution(
                [FindPackageShare('juggling_platform_2d_description'), 'rviz', 'dark.qss']
            ),
            description='RViz stylesheet',
        )
    )

    description_file = LaunchConfiguration('description_file')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    rviz_stylesheet = LaunchConfiguration('rviz_stylesheet')

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            description_file
        ]
    )

    robot_description_xacro = {
        'robot_description': ParameterValue(value=robot_description_content, value_type=str)
    }

    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='log',
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description_xacro],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file, '--stylesheet', rviz_stylesheet]
    )

    nodes_to_start = [
        joint_state_publisher_node,
        robot_state_publisher_node,
        rviz_node,
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)

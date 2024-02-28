from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import os

package_name = 'stereo_visual_odometry'

def generate_launch_description():

    stereo_node = Node(
        package=package_name,
        namespace='',
        executable='stereo_node_exec',
        name='stereo_node',
        output='screen'
    )

    return LaunchDescription([
        stereo_node,
    ])
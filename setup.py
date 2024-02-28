from setuptools import find_packages, setup

package_name = 'stereo_visual_odometry'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/svo.launch.py']),
        ('share/' + package_name, ['launch/mono.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bharadwajsirigadi',
    maintainer_email='bharadwajsirigadi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "stereo_node_exec = stereo_visual_odometry.svo:main",
            "mono_node_exec = stereo_visual_odometry.mono:main",
        ],
    },
)

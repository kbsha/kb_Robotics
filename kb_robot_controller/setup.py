from setuptools import find_packages, setup

package_name = 'kb_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "kb_test_node = kb_robot_controller.kb_first_node:main",
            "kb_pose_subscriber = kb_robot_controller.kb_pose_subscriber:main",
            "kb_draw_circle = kb_robot_controller.kb_draw_circle:main",
            "kb_turtle_controller = kb_robot_controller.kb_turtle_controller:main"
        ],
    },
)

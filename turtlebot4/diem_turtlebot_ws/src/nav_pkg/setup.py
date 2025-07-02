from setuptools import find_packages, setup

package_name = 'nav_pkg'

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
    maintainer='gdesimone',
    maintainer_email='44608428+gdesimone97@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "cone_detector_node_turtlebot = nav_pkg.cone_detector_turtlebot_node:main",
            "cone_passage_planner_node = nav_pkg.cone_passage_planner_node:main",
            "navigation_node = nav_pkg.navigation_node:main",
            "discovery_node = nav_pkg.discovery_node:main",
        ],
    },
)

from setuptools import find_packages, setup
from glob import glob
import os

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
            "cone_detection_node = nav_pkg.cone_detection_node:main",
            "pose_estimator_node = nav_pkg.pose_estimator_node:main",
            "navigation_node = nav_pkg.navigation_node:main"
        ],
    },
)

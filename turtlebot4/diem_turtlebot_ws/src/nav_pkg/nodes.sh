#!/bin/bash

# Terminale 1: Esecuzione del cone_detector_node
gnome-terminal --tab --title="Cone Detector" -- bash -c "echo 'EXECUTING CONE DETECTOR NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --symlink-install; source install/setup.bash; ros2 run nav_pkg cone_detector_node"

sleep 2

# Terminale 2: Esecuzione del cone_passage_planner
gnome-terminal --tab --title="Navigation" -- bash -c "echo 'EXECUTING NAVIGATION NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --symlink-install; source install/setup.bash; ros2 run nav_pkg navigation_node"

#!/bin/bash

# Terminale 1: Compilazione e esecuzione di camera_node
gnome-terminal --tab --title="Camera" -- bash -c "echo 'EXECUTING CAMERA NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg camera_node; sleep 5; exit"

sleep 2


# Terminale 2: Compilazione e esecuzione di discovery_node
gnome-terminal --tab --title="Discovery" -- bash -c "echo 'EXECUTING DISCOVERY NODE'; cd ~/turtlebot4/diem_turtlebot_ws;colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg discovery_node; sleep 5; exit"

sleep 2

# Terminale 3: Compilazione e esecuzione di navigation_node
gnome-terminal --tab --title="Navigation" -- bash -c "read -p 'Starting point x: ' start_x; read -p 'Starting point y: ' start_y; read -p 'Orientation (IN CAPS LOCK): ' orientation; echo 'EXECUTING NAVIGATION NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg navigation_node --x \$start_x --y \$start_y --orientation \$orientation; sleep 5; exit"


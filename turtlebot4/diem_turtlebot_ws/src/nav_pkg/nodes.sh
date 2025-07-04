#!/bin/bash

# Terminale 1: Esecuzione del cone_detector_node
gnome-terminal --tab --title="Cone Detector" -- bash -c "echo 'EXECUTING CONE DETECTOR NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg cone_detector_node_turtlebot"

sleep 2

# Terminale 2: Esecuzione del cone_passage_planner
gnome-terminal --tab --title="Cone Planner" -- bash -c "echo 'EXECUTING CONE PASSAGE PLANNER'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg cone_passage_planner"

sleep 2

# Terminale 3: Esecuzione del discovery_node (facoltativo, solo all’inizio)
gnome-terminal --tab --title="Discovery" -- bash -c "echo 'EXECUTING DISCOVERY NODE'; cd ~/turtlebot4/diem_turtlebot_ws; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg discovery_node"

sleep 2

# Terminale 4: Avvio del navigation_node con input dinamico
gnome-terminal --tab --title="Navigation" -- bash -c "
read -p 'Starting X: ' start_x;
read -p 'Starting Y: ' start_y;
read -p 'Starting Yaw: ' start_yaw;
read -p 'Goal X: ' goal_x;
read -p 'Goal Y: ' goal_y;
read -p 'Goal Yaw: ' goal_yaw;
cd ~/turtlebot4/diem_turtlebot_ws;
colcon build --packages-up-to nav_pkg;
source install/setup.bash;
ros2 run nav_pkg navigation_node --start_x \$start_x --start_y \$start_y --start_yaw \$start_yaw --goal_x \$goal_x --goal_y \$goal_y --goal_yaw \$goal_yaw"
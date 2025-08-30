#!/bin/bash

# Terminale 1: Esecuzione del cone_detection_node
gnome-terminal --tab --title="Cone Detector Node" -- bash -c "echo 'EXECUTING CONE TRACKING NODE'; cd /home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg cone_detection_node"

sleep 2

# Terminale 2: Esecuzione del navigation_node
gnome-terminal --tab --title="Navigation Node" -- bash -c "echo 'EXECUTING NAVIGATION NODE'; cd /home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg navigation_node"

sleep 2

# Terminale 3: Esecuzione del pose_estimator_node
gnome-terminal --tab --title="Pose Estimator Node" -- bash -c "echo 'EXECUTING POSE ESTIMATOR NODE'; cd /home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/; colcon build --packages-up-to nav_pkg; source install/setup.bash; ros2 run nav_pkg pose_estimator_node"
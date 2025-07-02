#!/bin/bash

# Terminale 1: Esecuzione di localization.launch.py
gnome-terminal --tab --title="Map" -- bash -c "echo 'LOADING MAP'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws/src/map; ros2 launch turtlebot4_navigation localization.launch.py map:=diem_map.yaml; sleep 5; exit"

# Terminale 2: Esecuzione di nav2.launch.py
gnome-terminal --tab --title="Nav Stack" -- bash -c "echo 'RUNNING NAVIGATION STACK'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws/src/map; ros2 launch turtlebot4_navigation nav2.launch.py; sleep 5; exit"

# Terminale 3: Esecuzione di view_robot.launch.py
gnome-terminal --tab --title="Rviz" -- bash -c "echo 'OPENING RVIZ'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws/src/map; ros2 launch turtlebot4_viz view_robot.launch.py; sleep 5; exit"


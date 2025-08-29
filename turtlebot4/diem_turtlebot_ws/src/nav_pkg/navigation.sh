#!/bin/bash

# Terminale 1: Localization con mappa
gnome-terminal --tab --title="Map" -- bash -c "echo 'LOADING MAP'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws/src/map/; ros2 launch turtlebot4_navigation localization.launch.py map:=diem_map.yaml"

sleep 2

# Terminale 2: Nav2 (navigation stack)
gnome-terminal --tab --title="Nav Stack" -- bash -c "echo 'RUNNING NAVIGATION STACK'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws; ros2 launch turtlebot4_navigation nav2.launch.py"

sleep 2

# Terminale 3: Rviz per visualizzazione
gnome-terminal --tab --title="Rviz" -- bash -c "echo 'OPENING RVIZ'; source ~/turtlebot4/diem_turtlebot_ws/install/setup.bash; cd ~/turtlebot4/diem_turtlebot_ws; ros2 launch turtlebot4_viz view_robot.launch.py"
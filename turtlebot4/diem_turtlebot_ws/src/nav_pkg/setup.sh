#!/bin/bash

sudo apt install –y ntpdate
sudo ntpdate ntp.unisa.it
sleep 1
ros2 topic list
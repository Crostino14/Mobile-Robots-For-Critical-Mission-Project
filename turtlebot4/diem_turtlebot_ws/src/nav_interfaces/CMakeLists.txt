cmake_minimum_required(VERSION 3.8)
project(nav_interfaces)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ConeDetection.msg"
  "msg/ConeDetectionArray.msg"
  DEPENDENCIES std_msgs
)

ament_package()
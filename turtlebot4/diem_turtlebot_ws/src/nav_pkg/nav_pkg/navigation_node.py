import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import json
import argparse

class NavigationNode(Node):
    def __init__(self, starting_point, start_orientation, goal_point, goal_orientation):
        super().__init__('navigation_node')
        self.navigator = TurtleBot4Navigator()
        self.goal_pose = self.navigator.getPoseStamped(goal_point, goal_orientation)
        
        initial_pose = self.navigator.getPoseStamped(starting_point, start_orientation)

        self.get_logger().info("Setting initial pose...")
        self.navigator.clearAllCostmaps()
        self.navigator.setInitialPose(initial_pose)

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()

        self.get_logger().info("Sending goal to Nav2...")
        self.navigator.goToPose(self.goal_pose)

        # Undock
        self.navigator.undock()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    
    with open("diem_turtlebot_ws/src/nav_pkg/nav_pkg/navigation_config.json", "r") as f:
        config = json.load(f)

    start_x = config["start"]["x"]
    start_y = config["start"]["y"]
    start_orientation = config["start"]["orientation"]
    goal_x = config["goal"]["x"]
    goal_y = config["goal"]["y"]
    goal_orientation = config["goal"]["orientation"]

    starting_point = [start_x, start_y]
    goal_point = [goal_x, goal_y]

    node = NavigationNode(starting_point, start_orientation, goal_point, goal_orientation)
    rclpy.spin(node)
    rclpy.shutdown()
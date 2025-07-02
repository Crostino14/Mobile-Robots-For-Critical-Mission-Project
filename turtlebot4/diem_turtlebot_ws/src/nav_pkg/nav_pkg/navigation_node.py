import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import math
import argparse

class NavigationNode(Node):
    def __init__(self, start_pose, goal_pose):
        super().__init__('navigation_node')
        self.navigator = TurtleBot4Navigator()
        self.goal_pose = goal_pose

        self.get_logger().info("Setting initial pose...")
        self.navigator.setInitialPose(start_pose)

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()

        self.get_logger().info("Sending goal to Nav2...")
        self.navigator.goToPose(goal_pose)

        # Pubblica anche su /final_goal per il planner
        self.final_goal_pub = self.create_publisher(PoseStamped, '/final_goal', 10)
        self.get_logger().info("Publishing final goal on /final_goal")
        self.final_goal_pub.publish(goal_pose)

def create_pose(x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = x
    pose.pose.position.y = y
    angle = math.radians(yaw_deg)
    pose.pose.orientation.z = math.sin(angle / 2.0)
    pose.pose.orientation.w = math.cos(angle / 2.0)
    return pose

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', type=float, required=True)
    parser.add_argument('--start_y', type=float, required=True)
    parser.add_argument('--start_yaw', type=float, default=0.0)
    parser.add_argument('--goal_x', type=float, required=True)
    parser.add_argument('--goal_y', type=float, required=True)
    parser.add_argument('--goal_yaw', type=float, default=0.0)
    args = parser.parse_args()

    start_pose = create_pose(args.start_x, args.start_y, args.start_yaw)
    goal_pose = create_pose(args.goal_x, args.goal_y, args.goal_yaw)

    node = NavigationNode(start_pose, goal_pose)
    rclpy.spin(node)
    rclpy.shutdown()
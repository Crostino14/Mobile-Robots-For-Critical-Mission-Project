import os
import argparse
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_pkg.graph import Graph
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from irobot_create_msgs.msg import KidnapStatus


class NavigationNode(Node):
    """
    ROS2 node for navigation using a graph structure.

    Attributes:
        graph (Graph): Instance of Graph containing the loaded graph structure.
        navigator (TurtleBot4Navigator): Instance of TurtleBot4Navigator for robot navigation.
        current_node (dict): Current robot position coordinates ('x' and 'y').
        current_orientation (str): Current robot orientation ('NORTH', 'EAST', 'SOUTH', 'WEST').
        current_direction (str): Current direction of movement ('STRAIGHTON', 'LEFT', 'RIGHT', 'GOBACK').
        previous_node (dict or None): Previous robot position coordinates if robot was kidnapped.
        previous_orientation (str or None): Previous robot orientation if robot was kidnapped.
        previous_direction (str or None): Previous direction of movement if robot was kidnapped.
        kidnap_status (bool): Flag indicating if the robot has been kidnapped.
    """    
    def __init__(self, graph_file_path, points_file_path, starting_point, orientation=None):
        """
        Initialize the navigation node.

        Args:
            graph_file_path (str): Path to the YAML file containing the graph structure.
            points_file_path (str): Path to the YAML file containing the points coordinates.
            starting_point (dict): Starting point with 'x' and 'y' keys representing coordinates.
            orientation (str, optional): Starting orientation ('NORTH', 'EAST', 'SOUTH', 'WEST').
        """
        super().__init__('navigation_node')

        # Kidnap topic QoS definition
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.graph = Graph(graph_file_path, points_file_path)
        self.navigator = TurtleBot4Navigator()

        self.current_node = starting_point
        self.current_orientation = orientation
        self.current_direction = "STRAIGHTON"

        self.previous_node = None
        self.previous_orientation = None
        self.previous_direction = None
        self.kidnap_status = False

        # Set initial position
        self.set_initial_pose(self.current_node, self.current_orientation)

        # Subscribe to the command topic
        self.command_subscription = self.create_subscription(
            String,
            'command',
            self.direction_callback,
            1
        )

        # Subscribe to the kidnap status topic
        self.kidnap_subscription = self.create_subscription(
            KidnapStatus,
            'kidnap_status',
            self.kidnap_callback,
            qos_profile
        )

        # Discovery publisher
        self.publisher = self.create_publisher(String, 'discovery', 1)

        # Perform navigation given a starting position, orientation and direction
        self.perform_navigation(self.current_node, self.current_orientation, self.current_direction)


    def direction_callback(self, msg):
        """
        Callback function for the 'command' topic.

        Publishes a discovery message if the received command is 'NOSIGNAL'. 
        Handles 'STOP' command by shutting down the node.
        Updates the current direction and initiates navigation for 'LEFT', 'RIGHT', 'STRAIGHTON', or 'GOBACK' commands.

        Args:
            msg (std_msgs.msg.String): Message containing the new direction command.
        """
        discovery_msg = String()
        discovery_msg.data = "START_DISCOVERY"

        if msg.data == "NOSIGNAL":
            self.publisher.publish(discovery_msg)
        else:
            if msg.data == "STOP":
                self.get_logger().info("Stop command received. Shutting down node.")
                rclpy.shutdown()
            elif msg.data == "LEFT" or msg.data == "RIGHT" or msg.data == "STRAIGHTON" or msg.data == "GOBACK":
                self.current_direction = msg.data
                self.perform_navigation(self.current_node, self.current_orientation, self.current_direction)


    def kidnap_callback(self, msg):
        """
        Callback function for the 'kidnap_status' topic.

        Handles changes in the robot's kidnap status based on received messages.
        If the robot is kidnapped (is_kidnapped is True), updates internal state and logs the event.
        If the robot is no longer kidnapped and was previously kidnapped, resets to the last known position
        and orientation before the kidnapping event, then performs navigation to continue its path.

        Args:
            msg (irobot_create_msgs.msg.KidnapStatus): Message containing the kidnap status.
        """

        # Access the is_kidnapped field from the KidnapStatus message
        is_kidnapped = msg.is_kidnapped

        if is_kidnapped:
            self.get_logger().info("robot has been kidnapped.")
            self.kidnap_status = True

        if not is_kidnapped and self.kidnap_status:
            self.kidnap_status = False
            self.get_logger().info("robot has been released.")
            # Reset the current position to the previous one
            self.current_node = self.previous_node
            self.current_orientation = self.previous_orientation
            self.current_direction = self.previous_direction

            self.previous_node = None
            self.previous_orientation = None
            self.previous_direction = None

            # Set the initial position after the robot has been kidnapped
            time.sleep(1.5)
            self.set_initial_pose(self.current_node, self.current_orientation)
            time.sleep(1.5)
            self.perform_navigation(self.current_node, self.current_orientation, self.current_direction)

    def perform_navigation(self, current_point, orientation, direction):
        """
        Perform navigation to the next node based on the current point, orientation, and direction.

        Finds the nearest node to the current point, calculates the next node and orientation 
        based on the current orientation and direction, navigates to the next node, and updates 
        the current and previous states.

        Args:
            current_point (dict): Current point with 'x' and 'y' keys representing coordinates.
            orientation (str): Current orientation as a string (e.g., 'NORTH', 'EAST').
            direction (str): Direction to move in as a string (e.g., 'STRAIGHTON').
        """
        # Find the nearest node to the starting point
        nearest_node = self.graph.find_nearest_point(current_point)
        self.get_logger().info(f"Starting from node: {nearest_node}")

        # Find the next node and the next orientation
        next_node, next_orientation = self.graph.get_next_node_and_orientation(
            orientation, direction, nearest_node
        )
        self.get_logger().info(f"Navigating to: {next_node}")
        self.get_logger().info(f"Whit direction: {direction}")

        # Navigate to the next node
        self.navigate_to_node(next_node, self.orientation_conversion(next_orientation))

        # Update the current state
        self.current_node = next_node
        self.current_orientation = next_orientation
        self.previous_node = current_point
        self.previous_orientation = orientation
        self.previous_direction = direction

    def set_initial_pose(self, start_point, orientation):
        """
        Set the initial pose of the robot.

        Calculates the initial pose based on the provided starting point and orientation,
        then sets this pose using the navigation system.

        Args:
            start_point (dict): Starting point with 'x' and 'y' keys representing coordinates.
            orientation (str): Starting orientation as a string (e.g., 'NORTH', 'EAST').
        """
        orientation = self.orientation_conversion(orientation)
        initial_pose = self.navigator.getPoseStamped([start_point['x'], start_point['y']], orientation)
        self.navigator.setInitialPose(initial_pose)

    def navigate_to_node(self, node, orientation):
        """
        Navigate to the specified node.

        Retrieves the coordinates of the node from the graph, computes the goal pose
        using the specified orientation, and initiates the navigation process to the node.

        Args:
            node (str): Node identifier to navigate to.
            orientation (TurtleBot4Directions): Orientation to maintain while navigating.
        """
        x, y = self.graph.get_coordinates(node)
        if x is not None and y is not None:
            goal_pose = self.navigator.getPoseStamped([x, y], orientation)
            self.navigator.startToPose(goal_pose)
            time.sleep(1.5)
        else:
            self.get_logger().error("Invalid node coordinates")

    def orientation_conversion(self, orientation):
        """
        Convert a string orientation to the corresponding TurtleBot4Directions value.

        Args:
            orientation (str): Orientation as a string (e.g., 'NORTH', 'EAST')

        Returns:
            TurtleBot4Directions: Corresponding TurtleBot4Directions value
        """
        if orientation == "NORTH":
            return TurtleBot4Directions.NORTH
        elif orientation == "EAST":
            return TurtleBot4Directions.EAST
        elif orientation == "SOUTH":
            return TurtleBot4Directions.SOUTH
        elif orientation == "WEST":
            return TurtleBot4Directions.WEST


def main(args=None):
    """
    Main function to initialize and spin the navigation node.
    """
    rclpy.init(args=args)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Navigation Node')
    parser.add_argument('--x', type=float, required=True, help='Starting x coordinate')
    parser.add_argument('--y', type=float, required=True, help='Starting y coordinate')
    parser.add_argument('--orientation', type=str, required=True, choices=['NORTH', 'EAST', 'SOUTH', 'WEST'], help='Starting orientation')
    args = parser.parse_args()

    starting_point = {'x': args.x, 'y': args.y}
    orientation = args.orientation

    # Get the absolute path to the current directory of the file
    package_path = os.path.dirname(os.path.abspath(__file__))

    # Find the path to the 'src' directory starting from the current directory of the file
    # Navigate up until you find the 'src' directory
    current_path = package_path
    while current_path != '/':
        src_path = os.path.join(current_path, 'src/nav_pkg/nav_pkg')
        if os.path.exists(src_path):
            break
        current_path = os.path.dirname(current_path)

    # Construct the paths to graph.yaml and points.yaml relative to the src_path
    graph_path = os.path.join(src_path, 'graph.yaml')
    points_path = os.path.join(src_path, 'points.yaml')

    navigation_node = NavigationNode(graph_path, points_path, starting_point, orientation)

    rclpy.spin(navigation_node)

    navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
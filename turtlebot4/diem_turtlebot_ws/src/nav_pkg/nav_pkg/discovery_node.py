import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor
from rclpy.clock import Duration
import threading,math

class DiscoveryNode(Node):
    """
    A ROS2 node for handling discovery and movement commands.

    This node initializes a ROS2 node named "Discovery_node" and sets up logging, a timer, a publisher,
    and a subscription to handle discovery messages. It manages the state of discovery and movement commands
    through various callbacks and methods.

    Attributes:
        cmd_pub (Publisher): Publisher for movement commands to the "/cmd_vel" topic.
        discovery_flag (bool): Flag indicating whether the discovery process is active.
        ok (bool): Flag indicating whether the node is ready to publish movement commands.
        discovery_subscription (Subscription): Subscription to the "discovery" topic.
    """
    def __init__(self):
        """
        Initializes the DiscoveryNode.

        This constructor initializes the node with the name "Discovery_node", sets the logging level to DEBUG,
        creates a timer to call the move_callback function periodically, sets up a publisher for the "/cmd_vel"
        topic, and subscribes to the "discovery" topic to handle discovery commands.
        """
        super().__init__("Discovery_node") 
        self.get_logger().set_level(LoggingSeverity.DEBUG)
        self.create_timer(0.1, self.move_callback) # Creating a timer

        self.discovery_flag = False
        self.ok = False
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10) 
        # Subscribe to discovery topic
        self.discovery_subscription = self.create_subscription(
            String,
            'discovery',
            self.discovery_callback,
            1
        )

    def discovery_callback(self, msg):
        """
        Callback function for handling discovery commands.

        This function is triggered when a message is received on the discovery topic.
        If the message data is "START_DISCOVERY" and the discovery process is not already running,
        it starts the discovery process in a new thread.

        :param msg: The message received from the discovery topic.
        :type msg: std_msgs.msg.String
        """    
        if msg.data == "START_DISCOVERY" and self.discovery_flag == False:
            threading.Thread(target=self.discovery).start() # Start discovery in a new thread

    def move_callback(self):
        """
        Callback function for publishing movement commands.

        This function publishes the current movement command if the system is in an operational state (`self.ok` is True).
        It sends the command message to the movement command publisher.

        :return: None
        """        
        if self.ok:
            self.cmd_pub.publish(self.msg)

        
    def rotate(self, angle_degrees, clockwise=True):
        """
        Rotates the robot by a specified angle.

        This function rotates the robot by a given angle in degrees. The direction of rotation
        can be specified as clockwise or counterclockwise. The rotation is executed by publishing
        angular velocity commands to the robot.

        :param angle_degrees: The angle to rotate the robot, in degrees.
        :type angle_degrees: float
        :param clockwise: Direction of rotation. True for clockwise, False for counterclockwise. Default is True.
        :type clockwise: bool
        :return: None
        """
        angle_radians = math.radians(angle_degrees)
        velocity = 1.57 # rad/s
        factor = -1 if clockwise else 1
        msg = Twist()
        msg.angular.z = velocity * factor
        self.msg = msg
        # Calculate the time to sleep based on the angle and velocity
        sleep_time = abs(angle_radians / velocity)
                
        self.ok = True
        
        self.sleep(sleep_time)
        
        self.ok = False

    def stop(self):
        """
        Stops the robot's movement.

        This function stops the robot by setting the linear and angular velocity to zero and publishing the command.
        The robot remains stationary for 0.5 seconds.

        :return: None
        """
        msg = Twist()
        msg.linear.x = 0.0 # m/s
        msg.angular.z = 0.0 #m/s
        self.msg = msg
        self.ok = True
        self.sleep(0.5)
        self.ok = False

    def sleep(self, time_seconds):
        """
        Sleeps the node for a specified duration.

        This function pauses the node's execution for a given number of seconds.

        :param time_seconds: The duration to sleep, in seconds.
        :type time_seconds: float
        :return: None
        """
        self.get_clock().sleep_for(Duration(seconds=time_seconds)) 

    def discovery(self):
        """
        Executes the discovery routine.

        This function performs a series of rotations to simulate a discovery routine. It rotates the robot
        in small increments, stops, and pauses between rotations. The routine is logged, and the discovery 
        flag is set to True during execution and reset to False upon completion.

        :return: None
        """
        self.discovery_flag = True
        self.get_logger().info("Discovery starting...")
        self.sleep(1)

        self.rotate(22.5, clockwise = False )
        self.stop()        
        self.sleep(1.5)
        self.rotate(22.5, clockwise = False )
        self.stop()        
        self.sleep(1.5)

        self.rotate(67.5, clockwise = True) 
        self.stop()        
        self.sleep(1.5)
        self.rotate(22.5, clockwise = True)
        self.stop()        
        self.sleep(1.5)

        self.rotate(45, clockwise = False )
        self.stop()        
        self.sleep(0.5)         
        self.stop()
        self.get_logger().info("Discovery ended.")
        self.discovery_flag = False

def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    discovery_node = DiscoveryNode()
    executor.add_node(discovery_node) # Adding node to executor
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    discovery_node.destroy_node()

if __name__ == "__main__":
    main()

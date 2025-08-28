import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from irobot_create_msgs.msg import KidnapStatus
from enum import Enum, auto
import json
import os
import numpy as np
import math
import statistics
import threading
from turtlebot4_navigation.turtlebot4_navigator import TaskResult, TurtleBot4Navigator
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor 
from nav2_msgs.srv import ClearEntireCostmap
import time

class NavState(Enum):
    GO_FINAL = auto()
    RECOVERY = auto()
    NAVIGATING_MID = auto()
    KIDNAP = auto()
    MISSION_COMPLETE = auto()

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # === PARAMS ===
        self.buffer_limit_max = 12
        self.buffer_limit_min = 4
        self.midpoint_buffer = []
        self.state = NavState.GO_FINAL

        self.final_goal = None
        self.current_goal = None
        self.current_robot_pose = None
        self.kidnap_state = False

        self.goal_merge_distance = 0.5
        self.outlier_std_threshold = 0.8
        self.min_pose_delta = 0.3  # metri
        self.min_yaw_delta = 10.0  # gradi

        self.state_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.buffer_lock = threading.Lock()

        self.costmap_service_busy = False

        self.qos_profile = QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        depth=10
                    )

        self._setup_subscriptions()
        self._setup_navigation()
        self._setup_timers()

        self.get_logger().info("Navigation node started in GO_FINAL")

    def _setup_subscriptions(self):
        """Setup essential ROS subscriptions."""
        # Sottoscrizione alle proposte di waypoint invece che ai coni
        self.create_subscription(
            PoseStamped,
            '/intermediate_waypoint',
            self._midpoint_callback,
            10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_pose_callback,
            10
        )

        self.create_subscription(
            KidnapStatus,
            '/kidnap_status',
            self._kidnap_status_callback,
            self.qos_profile
        )

    def _setup_navigation(self):
        """Initialize navigation system and set initial/final goal from config."""
        try:
            config_path = "/home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/src/nav_pkg/nav_pkg/navigation_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            start = config["start"]
            goal = config["goal"]

            self.navigator = TurtleBot4Navigator()
            
            # Imposta il goal finale
            self.final_goal = self.navigator.getPoseStamped([goal["x"], goal["y"]], goal["orientation"])
            self.get_logger().info(f"Goal finale caricato: ({goal['x']}, {goal['y']})")

            # Imposta la posa iniziale
            initial_pose = self.navigator.getPoseStamped([start["x"], start["y"]], start["orientation"])
            self.navigator.setInitialPose(initial_pose)
            self.get_logger().info("Posizione iniziale inviata.")

            if self.navigator.getDockedStatus():
                self.navigator.undock()

            self.navigator.clearAllCostmaps()
            self.navigator.waitUntilNav2Active()

        except Exception as e:
            self.get_logger().error(f"Navigation setup failed: {e}")

    def _setup_timers(self):
        """Setup periodic timers."""
        # Timer principale per la macchina a stati
        self.create_timer(0.5, self.fsm_step)  # 2Hz FSM loop
        
        # Timer per l'elaborazione delle proposte in background
        #self.create_timer(1.0, self._midpoint_callback) # 1Hz

    def _midpoint_callback(self, msg):
        
        with self.buffer_lock:
            if len(self.midpoint_buffer) >= self.buffer_limit_max:
                self.midpoint_buffer.pop(0)
            self.midpoint_buffer.append(msg)
        self.get_logger().info(f"Midpoint received. Buffer size: {len(self.midpoint_buffer)}")

    def _amcl_pose_callback(self, msg):
        """Thread-safe pose update"""
        with self.pose_lock:
            self.current_robot_pose = msg.pose.pose

    def _kidnap_status_callback(self, msg):
        """Thread-safe kidnap status update"""
        with self.state_lock:
            self.kidnap_state = True if msg.is_kidnapped else False

    def fsm_step(self):
        with self.state_lock:
            if self.state == NavState.GO_FINAL:
                self._on_final_goal()

            if self.state == NavState.RECOVERY:
                self._on_recovery()

            elif self.state == NavState.NAVIGATING_MID:
                self._on_navigating_mid()

            elif self.state == NavState.KIDNAP:
                self._on_kidnap()

            elif self.state == NavState.MISSION_COMPLETE:
                self._on_mission_complete()

    def _on_final_goal(self):
        self.get_logger().info("Final goal state")
        if self.final_goal is None:
            self.get_logger().error("Final goal not set. Cannot proceed.")
            return

        self.get_logger().info(f"Navigating to final goal: {self.final_goal.pose.position}")
        self.navigator.goToPose(self.final_goal)
        
        result = self.navigator.getResult()
        # Se il task viene completato con successo vado al mission complete
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Reached final goal successfully.")
            self.state = NavState.MISSION_COMPLETE
        else:
            self.state = NavState.RECOVERY

    def _on_mission_complete(self):
        self.get_logger().info("Mission complete state. Shutting down.")
        self.midpoint_buffer.clear()
        self.destroy_node()
        rclpy.shutdown()

    def _on_recovery(self):
        self.get_logger().info("Recovery state")
        self.get_logger().info(f"Lunghezza buffer waypoint {len(self.midpoint_buffer)}")
        # Se il turtlebot è stato rapito
        if self.kidnap_detected():
            self.get_logger().warn("Kidnap detected during navigation to midpoints.")
            self.state = NavState.KIDNAP
        # Se il buffer dei waypoint intermedi è pieno, vado in recovery
        elif len(self.midpoint_buffer) >= self.buffer_limit_min:
            self.state = NavState.NAVIGATING_MID
        else:
            self.state = NavState.GO_FINAL
            
    # PER SETTEMBRE: sta funzione non ha un punto di uscita se non vi è alcun goal intermedio da passare? Usciamo solo se stavamo andando a un goal intermedio e siamo a una certa distanza da esso in questo momento. Forse per risolvere potremmo controllare, dopo l'if avg_pose is not None, se il self.current_goal è None. In questo caso torniamo al recovery (soluzione buttata così)
    
    def _on_navigating_mid(self):
        self.get_logger().info("Navigating midpoints state")

        # Calcola la media filtrata dei waypoint
        avg_pose = self.get_filtered_average_pose(self.midpoint_buffer)

        dist = 1000
        dx, dy = None, None

        with self.pose_lock:
            if self.current_goal is not None:
                dx = self.current_goal.pose.position.x - self.current_robot_pose.position.x
                dy = self.current_goal.pose.position.y - self.current_robot_pose.position.y

        if dx and dy:
            dist = math.hypot(dx, dy)
        
        # PER SETTEMBRE: questa soglia delle distanza evitava la rotazione del robot dopo la mid pose, soglie minori non funzionano

        if dist <= 0.5 and not self.navigator.isTaskComplete() and self.current_goal is not None:
            self.navigator.cancelTask()

            with self.buffer_lock:
                self.midpoint_buffer.clear()
            
            self.current_goal = None

            self.get_logger().info("Reached average pose successfully.")
            self.state = NavState.RECOVERY
            
            return
        
        # Se non è sufficientemente diverso, continuiamo a seguire quello corrente
        self.get_logger().info(f"\nPrima del confronto\n- current : {self.current_goal}\n- avg_pose : {avg_pose}")
        if self.current_goal is not None and avg_pose is not None and not self.is_pose_significantly_different(avg_pose, self.current_goal):
            self.get_logger().info("Average pose similar to current goal. Continuing with current goal.")
            return

        if avg_pose is not None:
            self.get_logger().info(f"Navigating to average pose: {avg_pose.pose.position}")
            self.current_goal = avg_pose
            self.navigator.goToPose(avg_pose)
            
            """with self.buffer_lock:
                self.midpoint_buffer.clear()"""

        # POSSIAMO ANCHE METTERE DIRETTAMENTE ELSE RECOVERY

    def _on_kidnap(self):
        # Logica per gestire il kidnap da rivedere quando la implementazione sarà più chiara
    
        self.state = NavState.GO_FINAL

    def kidnap_detected(self):
        """Check if a kidnap event has occurred."""
        with self.pose_lock:
            if self.kidnap_state:
                self.get_logger().warn("Kidnap detected!")
                return True
            else:
                return False

    def is_pose_significantly_different(self, pose1: PoseStamped, pose2: PoseStamped):
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        dist = math.hypot(dx, dy)

        yaw1 = self.extract_yaw(pose1)
        yaw2 = self.extract_yaw(pose2)
        dyaw = abs(math.degrees(yaw1 - yaw2)) % 360
        if dyaw > 180:
            dyaw = 360 - dyaw

        return dist > self.min_pose_delta or dyaw > self.min_yaw_delta

    def extract_yaw(self, pose_stamped):
        """Estrai yaw da un quaternion."""
        q = pose_stamped.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def compute_orientation_towards(self, current_pose, target_pose):
        """Calcola il quaternion orientato verso il target_pose."""
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        yaw = (math.degrees(math.atan2((dy), (dx)) + 360)) % 360
        return yaw

    def get_filtered_average_pose(self, waypoints):
        """Calcola la media filtrata dei waypoint, scartando outlier."""
        if len(waypoints) <= self.buffer_limit_min:
            return None
        x_vals = [wp.pose.position.x for wp in waypoints]
        y_vals = [wp.pose.position.y for wp in waypoints]

        mean_x = np.mean(x_vals)
        mean_y = np.mean(y_vals)
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)

        filtered = [
            wp for wp in waypoints
            if np.linalg.norm([
                wp.pose.position.x - mean_x,
                wp.pose.position.y - mean_y
            ]) < self.outlier_std_threshold * math.hypot(std_x, std_y)
        ]

        filtered = [wp for wp in waypoints if np.linalg.norm([wp.pose.position.x - mean_x, wp.pose.position.y - mean_y]) < self.goal_merge_distance]
        if not filtered:
            return self.current_goal
        avg_x = round(np.mean([wp.pose.position.x for wp in filtered]), 2)
        avg_y = round(np.mean([wp.pose.position.y for wp in filtered]), 2)

        avg_pose = PoseStamped()
        avg_pose.header.frame_id = "map"
        avg_pose.header.stamp = self.get_clock().now().to_msg()
        avg_pose.pose.position.x = avg_x
        avg_pose.pose.position.y = avg_y

        # Orientamento verso la media
        current_pose = self._get_current_pose_stamped()
        avg_yaw = self.compute_orientation_towards(current_pose, avg_pose)
        avg_pose = self.navigator.getPoseStamped([avg_x, avg_y], avg_yaw)
        return avg_pose

    def _get_current_pose_stamped(self):
        """Restituisce la pose corrente come PoseStamped."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        with self.pose_lock:
            if self.current_robot_pose is not None:
                pose.pose = self.current_robot_pose
        return pose
    
def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
Project: Mobile Robots for Critical Mission — nav_pkg.navigation_node

Student and Creator:

Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
Chiara Ferraioli         0622702169      c.ferraioli30@studenti.unisa.it
Asja Antonucci           0622702437      a.antonucci5@studenti.unisa.it

Purpose:
This module implements `NavigationNode`, a ROS2 node that implements a
small finite-state controller responsible for long-distance navigation to a
final mission goal while accepting intermediate waypoints from the perception
stack.

Design summary and responsibilities:
- Subscribe to `/intermediate_waypoint` (geometry_msgs/PoseStamped) for
    candidate midpoints produced by the pose estimator node.
- Subscribe to `/amcl_pose` (PoseWithCovarianceStamped) to obtain the robot
    estimate used for local decisions such as heading corrections and distance
    computations.
- Subscribe to `/kidnap_status` (irobot_create_msgs/KidnapStatus) and enter
    a dedicated KIDNAP state while the robot's localisation is considered
    invalid.
- Maintain a small FIFO buffer of midpoints, compute a robust averaged
    waypoint using simple outlier rejection, and command the `TurtleBot4Navigator`
    to visit intermediate poses when appropriate.

Configuration and assumptions:
- A JSON configuration file `navigation_config.json` is read from the
    package directory to obtain a `start` and `goal` pose for the mission.
- The `TurtleBot4Navigator` wrapper is expected to provide helpers such as
    `getPoseStamped()`, `goToPose()`, `setInitialPose()`, `cancelTask()`,
    `clearAllCostmaps()`, `waitUntilNav2Active()` and `spin()`.
- The node assumes AMCL and Nav2 are present in the runtime environment and
    that `/amcl_pose` is published regularly while localisation is valid.

How to run:
1. Source your ROS 2 and workspace environment.
2. Ensure that AMCL and Nav2 are running and that the pose estimator and
    cone detector (if used) are publishing intermediate waypoints.
3. Run:
    ros2 run nav_pkg navigation_node
"""

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor 
from nav2_msgs.srv import ClearEntireCostmap
import time

class NavState(Enum):
        """Finite states for the navigation state machine.

        - GO_FINAL: drive to the final mission goal configured in JSON.
        - RECOVERY: attempt to recover from a failed navigation action or after
            completing an intermediate waypoint.
        - NAVIGATING_MID: actively navigating towards a filtered intermediate
            waypoint computed from the midpoint buffer.
        - KIDNAP: detected kidnap (AMCL pose reset) — pause and perform recovery
            spin manoeuvres before resuming.
        - MISSION_COMPLETE: final goal reached; the node will shut down.
        """

        GO_FINAL = auto()
        RECOVERY = auto()
        NAVIGATING_MID = auto()
        KIDNAP = auto()
        MISSION_COMPLETE = auto()

class NavigationNode(Node):
    def __init__(self):
        """Initialise the NavigationNode.

        The constructor initialises configuration parameters, thread locks
        used to protect shared state when callbacks run concurrently, the
        navigator wrapper and ROS subscriptions/timers.

        Important attributes (summary):
        - **midpoint_buffer**: FIFO list of incoming intermediate waypoints.
        - **buffer_limit_min/_max**: sizing thresholds for switching states.
        - **final_goal**: PoseStamped produced from the JSON configuration.
        - **current_goal**: PoseStamped currently being sent to the navigator.
        - **current_robot_pose**: latest AMCL pose (protected by `pose_lock`).
        - **state**: current FSM state from NavState.
        """

        super().__init__('navigation_node')

        # === Buffer and state thresholds ===
        # Maximum number of intermediate waypoints to keep in memory. When
        # the buffer becomes full, the oldest entry is discarded.
        self.buffer_limit_max = 12
        # Minimum number of proposals required to consider computing an
        # averaged intermediate waypoint.
        self.buffer_limit_min = 4
        # Container storing incoming `PoseStamped` objects published by
        # the pose estimator node; treated as a FIFO.
        self.midpoint_buffer = []

        # Start in the default 'drive to final goal' state.
        self.state = NavState.GO_FINAL

        # === Navigation goals & robot pose ===
        self.final_goal = None
        self.current_goal = None
        # Latest AMCL pose message (PoseWithCovarianceStamped.pose)
        self.current_robot_pose = None
        self.kidnap_state = False

        # === Filtering / merging parameters ===
        # Distance within which proposals are merged (metres).
        self.goal_merge_distance = 0.5
        # Threshold used to reject spatial outliers (standard-deviation based).
        self.outlier_std_threshold = 0.8
        # Minimum translation and yaw difference to treat two poses as
        # meaningfully different.
        self.min_pose_delta = 0.3  # metres
        self.min_yaw_delta = 10.0  # degrees

        # === Threading locks ===
        # Separate locks reduce contention between callbacks.
        self.state_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.kidnap_lock = threading.Lock()
        self.buffer_lock = threading.Lock()

        # Small flag used when interacting with costmap services (kept for
        # potential future expansions).
        self.costmap_service_busy = False

        # === QoS profiles ===
        # Reliable profile for critical topics.
        self.qos_profile = QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        depth=10
                    )
        # Best-effort and volatile profile for rapidly-changing state such
        # as 'kidnap' status where late messages are not important.
        self.qos_best_effort_volatile = QoSProfile(
            depth=1, 
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE 
        )

        # Wire up subscriptions, navigator initialisation and periodic
        # timers. These helper methods perform I/O and therefore are kept
        # outside the constructor body for clarity.
        self._setup_subscriptions()
        self._setup_navigation()
        self._setup_timers()

        self.get_logger().info("Navigation node started in GO_FINAL")

    def _setup_subscriptions(self):
                """Set up essential ROS subscriptions.

                Subscriptions established:
                - `/intermediate_waypoint` (PoseStamped): accepted intermediate
                    waypoints computed by the pose estimator node.
                - `/amcl_pose` (PoseWithCovarianceStamped): robot pose estimate used
                    for local decisions and orientation calculations.
                - `/kidnap_status` (irobot_create_msgs/KidnapStatus): best-effort
                    updates reporting that the localisation has been reset.

                The subscriptions use conservative QoS profiles; the kidnap status
                uses a best-effort / volatile profile because late messages are not
                helpful and should not be queued.
                """

                # Subscribe to intermediate waypoint proposals instead of raw cone detections
                self.create_subscription(
                        PoseStamped,
                        '/intermediate_waypoint',
                        self._midpoint_callback,
                        10
                )

                # AMCL pose - used to compute distances and orientations for local
                # decisions. Reliable delivery is preferred for pose updates.
                self.create_subscription(
                        PoseWithCovarianceStamped,
                        '/amcl_pose',
                        self._amcl_pose_callback,
                        10
                )

                # Kidnap status - rapid, volatile updates; late messages are ignored
                # and do not need storing in a queue.
                self.create_subscription(
                        KidnapStatus,
                        '/kidnap_status',
                        self._kidnap_status_callback,
                        self.qos_best_effort_volatile
                )

    def _setup_navigation(self):
        """Initialise the TurtleBot4 navigator and read mission config.

        Behaviour:
        - Load the JSON configuration file `navigation_config.json` which must
          contain `start` and `goal` entries with x, y and orientation.
        - Create a `TurtleBot4Navigator` instance used to interact with Nav2.
        - Send the initial pose to AMCL and set the final mission goal.
        - If the robot is docked, attempt to undock before proceeding.
        - Clear costmaps and wait for Nav2 to become active.

        Note: This function logs exceptions rather than raising them so that
        the node will still exist for debugging even if the configuration
        file is missing or malformed.
        """
        try:
            config_path = "/home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/src/nav_pkg/nav_pkg/navigation_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            start = config["start"]
            goal = config["goal"]

            # Navigator is a thin wrapper around Nav2 calls used by the
            # assignment. It provides convenience helpers for commonly used
            # actions such as goToPose(), setInitialPose(), cancelTask() etc.
            self.navigator = TurtleBot4Navigator()
            
            # Set the mission final goal using the helper. The navigator
            # returns a properly formed PoseStamped in the map frame.
            self.final_goal = self.navigator.getPoseStamped([goal["x"], goal["y"]], goal["orientation"])
            self.get_logger().info(f"Final mission goal loaded: ({goal['x']}, {goal['y']})")

            # Send initial pose to AMCL so that localisation starts with the
            # expected initial estimate.
            initial_pose = self.navigator.getPoseStamped([start["x"], start["y"]], start["orientation"])
            self.navigator.setInitialPose(initial_pose)
            self.get_logger().info("Initial robot pose published to AMCL.")

            # If docked, perform an undock manoeuvre before clearing costmaps
            # and waiting for Nav2 readiness.
            if self.navigator.getDockedStatus():
                self.navigator.undock()

            # Ensure costmaps are clean and Nav2 is available before we
            # attempt to send any navigation requests.
            self.navigator.clearAllCostmaps()
            self.navigator.waitUntilNav2Active()

        except Exception as e:
            # Logging only: remain operational to inspect the issue from
            # logs rather than failing with an unhandled exception.
            self.get_logger().error(f"Navigation setup failed: {e}")

    def _setup_timers(self):
        """Setup periodic timers used by the node.
        
        Currently the node uses a single timer which executes the finite
        state machine at 2 Hz. Additional timers can be enabled (commented
        out) to periodically process incoming proposals in a separate
        background loop.
        """
        # Timer principale per la macchina a stati
        self.create_timer(0.5, self.fsm_step)  # 2Hz FSM loop
        
        # Timer per l'elaborazione delle proposte in background
        #self.create_timer(1.0, self._midpoint_callback) # 1Hz

    def _midpoint_callback(self, msg):
        """Callback that receives intermediate waypoints.

        The function inserts each incoming PoseStamped into a FIFO buffer
        guarded by `buffer_lock`. When the buffer exceeds
        `buffer_limit_max` the oldest entry is removed. This design prefers
        recent proposals while still allowing a small temporal window to be
        aggregated by the averaging routine.
        """
        with self.buffer_lock:
            if len(self.midpoint_buffer) >= self.buffer_limit_max:
                # Discard the oldest proposal to keep the buffer size bounded.
                self.midpoint_buffer.pop(0)
            self.midpoint_buffer.append(msg)
        self.get_logger().info(f"Midpoint received. Buffer size: {len(self.midpoint_buffer)}")

    def _amcl_pose_callback(self, msg):
        """Thread-safe AMCL pose update.

        The node stores the latest `PoseWithCovarianceStamped.pose` object in
        `self.current_robot_pose`. The attribute is protected by
        `pose_lock` because other threads (timers or subscriptions) read it.
        """
        with self.pose_lock:
            self.current_robot_pose = msg.pose

    def _kidnap_status_callback(self, msg):
        """Thread-safe update of kidnap/localisation-reset status.

        The `KidnapStatus` message originates from the Create SDK. When a
        kidnap is reported the node immediately transitions the FSM to the
        `KIDNAP` state under the protection of `state_lock`.
        """
        with self.kidnap_lock:
            self.kidnap_state = True if msg.is_kidnapped else False

        if msg.is_kidnapped:
            with self.state_lock:
                self.state = NavState.KIDNAP

    def fsm_step(self):
        """Single tick of the finite state machine.

        This method is executed from the timer callback. It acquires
        `state_lock` and dispatches to the handler corresponding to the
        current NavState. Handlers are responsible for updating `self.state`
        to effect transitions.
        """
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
        """Handler executed when the FSM is in GO_FINAL.

        Behaviour:
        - If no final goal is configured, log an error and return.
        - Command the navigator to drive to the final goal and await the
          result. On success transition to MISSION_COMPLETE; otherwise go
          to RECOVERY to attempt intermediate navigation/replanning.
        """
        self.get_logger().info("Final goal state")
        if self.final_goal is None:
            self.get_logger().error("Final goal not set. Cannot proceed.")
            return

        self.get_logger().info(f"Navigating to final goal: {self.final_goal.pose.position}")
        self.navigator.goToPose(self.final_goal)
        
        result = self.navigator.getResult()
        # If the navigation task completed successfully, mark mission done.
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Reached final goal successfully.")
            self.state = NavState.MISSION_COMPLETE
        else:
            # Non-successful outcomes are handled via the recovery state.
            self.state = NavState.RECOVERY

    def _on_mission_complete(self):
        """Handler for mission completion.

        Clears buffers and performs a clean shutdown of the node and rclpy.
        """
        self.get_logger().info("Mission complete state. Shutting down.")
        self.midpoint_buffer.clear()
        self.destroy_node()
        rclpy.shutdown()

    def _on_recovery(self):
        """Recovery handler.

        Recovery decides whether to attempt to navigate intermediate
        waypoints (if enough proposals exist), transition to the kidnap
        handler if a kidnap is detected, or resume driving to the final
        goal.
        """
        self.get_logger().info("Recovery state")
        self.get_logger().info(f"Intermediate waypoint buffer length: {len(self.midpoint_buffer)}")
        # If a kidnap is detected transition immediately to the KIDNAP state.
        if self.kidnap_detected():
            self.get_logger().warn("Kidnap detected during navigation to midpoints.")
            self.state = NavState.KIDNAP
        # If we have enough intermediate proposals, attempt to navigate
        # towards their filtered average.
        elif len(self.midpoint_buffer) >= self.buffer_limit_min:
            self.state = NavState.NAVIGATING_MID
        else:
            # Otherwise resume navigation to the final goal.
            self.state = NavState.GO_FINAL
            
    # PER SETTEMBRE: sta funzione non ha un punto di uscita se non vi è alcun goal intermedio da passare? Usciamo solo se stavamo andando a un goal intermedio e siamo a una certa distanza da esso in questo momento. Forse per risolvere potremmo controllare, dopo l'if avg_pose is not None, se il self.current_goal è None. In questo caso torniamo al recovery (soluzione buttata così)
    
    def _on_navigating_mid(self):
        self.get_logger().info("Navigating midpoints state")

        # Compute a filtered average from proposals in the buffer.
        avg_pose = self.get_filtered_average_pose(self.midpoint_buffer)

        # Default to a large distance; will be overwritten below if possible.
        dist = 1000
        dx, dy = None, None

        # Compute distance to the current goal using the most recent AMCL
        # pose. Both reads are protected by `pose_lock`.
        with self.pose_lock:
            if self.current_goal is not None and self.current_robot_pose is not None:
                dx = self.current_goal.pose.position.x - self.current_robot_pose.pose.position.x
                dy = self.current_goal.pose.position.y - self.current_robot_pose.pose.position.y

        if dx and dy:
            dist = math.hypot(dx, dy)
        
        # Historically this distance threshold prevented undesired rotation
        # after arriving at a mid pose; thresholds lower than 0.5 did not
        # behave well in early experiments.

        if dist <= 0.5 and not self.navigator.isTaskComplete() and self.current_goal is not None:
            # Cancel the current task and treat the average waypoint as
            # reached. Clear the buffer to remove stale proposals.
            self.navigator.cancelTask()

            with self.buffer_lock:
                self.midpoint_buffer.clear()
            
            self.current_goal = None

            self.get_logger().info("Reached average pose successfully.")
            self.state = NavState.RECOVERY
            
            return
        
        # If the computed average is not meaningfully different from the
        # current goal then continue with the current goal to avoid
        # frequent re-planning.
        self.get_logger().info(f"\nBefore comparison\n- current : {self.current_goal}\n- avg_pose : {avg_pose}")
        if self.current_goal is not None and avg_pose is not None and not self.is_pose_significantly_different(avg_pose, self.current_goal):
            self.get_logger().info("Average pose similar to current goal. Continuing with current goal.")
            return

        # If we have an average pose that differs sufficiently, command the
        # navigator to go to this new intermediate waypoint.
        if avg_pose is not None:
            self.get_logger().info(f"Navigating to average pose: {avg_pose.pose.position}")
            self.current_goal = avg_pose
            self.navigator.goToPose(avg_pose)
            
            """with self.buffer_lock:
                self.midpoint_buffer.clear()"""

    def _on_kidnap(self):
        """Handle the KIDNAP state.

        When a kidnap event resolves (i.e. the kidnap flag is cleared) the
        node attempts to re-orient the robot towards the final goal and then
        resumes normal recovery behaviour. While the kidnap flag is set the
        node ensures the robot is stopped and clears buffers/costmaps.

        The re-orientation procedure:
        1. Compute current yaw and desired yaw towards the final goal.
        2. Normalise the angle difference to (-pi, pi] and use the
           navigator.spin() helper to rotate in place.
        3. Do a small secondary spin to improve final heading stability.
        """
        if not self.kidnap_detected():
            self.get_logger().info("Kidnap resolved. Resuming navigation.")
            
            # Clear any active goals and buffers before attempting to resume.
            self.current_goal = None
            self.midpoint_buffer.clear()
            self.navigator.clearAllCostmaps()
            self.navigator.cancelTask()
            
            # Compute yaw correction relative to the final goal.
            yaw = math.radians(math.degrees(self.extract_yaw(self.current_robot_pose) + 360) % 360)
            goal_yaw = math.radians(self.compute_orientation_towards(self.current_robot_pose, self.final_goal))
            normalized_yaw = self._normalize_angle(goal_yaw - yaw)
            
            self.get_logger().info(f"Current yaw: {(math.degrees(yaw)):.2f}, Goal yaw: {math.degrees(goal_yaw):.2f}, Rotation needed: {math.degrees(normalized_yaw):.2f}")

            # Allow sensors and controllers to settle briefly.
            time.sleep(1.0)
            
            # Rotate the robot by the required yaw amount (in radians).
            self.navigator.spin(spin_dist=normalized_yaw, time_allowance=20)
            
            with self.buffer_lock:
                self.midpoint_buffer.clear()
            
            # A secondary corrective spin to improve heading stability. The
            # sign is chosen to reduce large residual errors.
            second_spin = -math.copysign(math.radians(60), normalized_yaw)

            time.sleep(1.0)
            self.navigator.spin(spin_dist=second_spin, time_allowance=20)

            self.state = NavState.RECOVERY
        
        else:
            # While still kidnapped ensure navigation is stopped and costmaps
            # are cleared so the robot doesn't attempt unsafe motions.
            self.get_logger().info("Handling kidnap event.")

            self.current_goal = None
            self.midpoint_buffer.clear()
            self.navigator.clearAllCostmaps()
            self.navigator.cancelTask()

    def kidnap_detected(self):
        """Return True if a kidnap event is currently recorded.

        The function reads the flag under `kidnap_lock` and logs a warning
        at the moment a kidnap is recognised.
        """
        with self.kidnap_lock:
            if self.kidnap_state:
                self.get_logger().warn("Kidnap detected!")
                return True
            else:
                return False

    def is_pose_significantly_different(self, pose1: PoseStamped, pose2: PoseStamped):
        """Decide whether two PoseStamped objects are meaningfully different.

        Comparison is performed using Euclidean distance in the XY plane and
        the absolute angular difference in yaw. The yaw difference is
        wrapped to the [0, 180] range for a consistent comparison.

        Returns True if either the translation or rotation difference exceeds
        the configured thresholds `min_pose_delta` or `min_yaw_delta`.
        """
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
        """Extract yaw (rotation about Z) from a PoseStamped quaternion.

        The function returns the yaw angle in radians using the standard
        conversion from a quaternion.
        """
        q = pose_stamped.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """Normalise an angle to the interval (-pi, pi].

        This helper is used to compute minimal rotation distances when
        aligning headings. The implementation is numerically stable for
        typical robot heading values.
        """
        a = (angle + math.pi) % (2.0 * math.pi)
        if a <= 0.0:
            a += 2.0 * math.pi
        return a - math.pi

    def compute_orientation_towards(self, current_pose, target_pose):
        """Compute the heading (in degrees) from current_pose towards target_pose.

        The returned value is the yaw angle in degrees in the range [0,360).
        This value is convenient when interacting with helper functions that
        accept rotation in degrees.
        """
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        yaw = (math.degrees(math.atan2((dy), (dx)) + 360)) % 360
        return yaw

    def get_filtered_average_pose(self, waypoints):
        """Compute a robust average PoseStamped from a list of waypoints.

        Steps:
        1. If the buffer is too small return None (insufficient proposals).
        2. Compute the mean and standard deviation across X and Y.
        3. Remove spatial outliers using a threshold proportional to the
           combined standard deviation.
        4. Additionally ensure that remaining points fall within
           `goal_merge_distance` of the computed mean.
        5. Return a `PoseStamped` built from the averaged X/Y and with an
           orientation computed to face the averaged point from the robot's
           current pose.

        The returned pose is in the `map` frame. If filtering removes all
        points the method conservatively returns `self.current_goal`.
        """
        if len(waypoints) <= self.buffer_limit_min:
            return None
        x_vals = [wp.pose.position.x for wp in waypoints]
        y_vals = [wp.pose.position.y for wp in waypoints]

        mean_x = np.mean(x_vals)
        mean_y = np.mean(y_vals)
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)

        # First-stage outlier rejection: spatial distance compared to a
        # scaled combination of standard deviations.
        filtered = [
            wp for wp in waypoints
            if np.linalg.norm([
                wp.pose.position.x - mean_x,
                wp.pose.position.y - mean_y
            ]) < self.outlier_std_threshold * math.hypot(std_x, std_y)
        ]

        # Second-stage: enforce a hard merge distance from the computed mean
        filtered = [wp for wp in filtered if np.linalg.norm([wp.pose.position.x - mean_x, wp.pose.position.y - mean_y]) < self.goal_merge_distance]
        if not filtered:
            # If no proposals survived conservative filtering, continue to
            # follow the current goal to avoid oscillation.
            return self.current_goal
        avg_x = round(np.mean([wp.pose.position.x for wp in filtered]), 2)
        avg_y = round(np.mean([wp.pose.position.y for wp in filtered]), 2)

        avg_pose = PoseStamped()
        avg_pose.header.frame_id = "map"
        avg_pose.header.stamp = self.get_clock().now().to_msg()
        avg_pose.pose.position.x = avg_x
        avg_pose.pose.position.y = avg_y

        # Compute an orientation facing the averaged position relative to
        # the robot's current pose and obtain a properly formed PoseStamped
        # via the navigator helper.
        current_pose = self._get_current_pose_stamped()
        avg_yaw = self.compute_orientation_towards(current_pose, avg_pose)
        avg_pose = self.navigator.getPoseStamped([avg_x, avg_y], avg_yaw)
        return avg_pose

    def _get_current_pose_stamped(self):
        """Return the most recent robot pose as a `PoseStamped` in `map`.

        If no AMCL pose has been received yet the returned PoseStamped will
        contain an empty pose body but the header will still be populated
        with the current time and frame id.
        """
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        with self.pose_lock:
            if self.current_robot_pose is not None:
                pose.pose = self.current_robot_pose.pose
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
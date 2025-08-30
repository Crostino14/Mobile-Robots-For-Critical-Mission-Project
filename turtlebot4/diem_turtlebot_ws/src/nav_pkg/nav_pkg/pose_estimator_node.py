"""
Project: Mobile Robots for Critical Mission — nav_pkg.pose_estimator_node

Student and Creator:

Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
Chiara Ferraioli         0622702169      c.ferraioli30@studenti.unisa.it
Asja Antonucci           0622702437      a.antonucci5@studenti.unisa.it

Purpose:
This module implements `PoseEstimatorNode`, a ROS2 node that consumes
`nav_interfaces/ConeDetectionArray` messages produced by the cone detector
and computes an intermediate waypoint (map-frame `PoseStamped`) that the
navigation stack can use to navigate around cones.

Design summary and responsibilities:
- Subscribe to `/detected_cones` (cone detections), `/oakd/stereo/camera_info`
    (camera intrinsics) and `/amcl_pose` (robot pose estimate).
- Convert cone pixel coordinates and reported depth into camera-frame 3D
    points, then transform them to the `map` frame using TF2.
- For a pair of aligned cones compute their midpoint in the map frame.
- For a single cone compute a laterally offset intermediate point to pass
    the cone on the correct side (red → right/left convention implemented
    in the node).
- Publish the computed waypoint as a `PoseStamped` on
    `/intermediate_waypoint`.

Configuration and assumptions:
- Camera intrinsics are read from `/oakd/stereo/camera_info` and stored on
    first message reception. The node assumes those intrinsics are correct.
- Cone depths are provided by the cone detector (`ConeDetection.depth`) in
    metres. The node performs additional heuristic adjustments for large
    distances.
- TF between the camera optical frame (`oakd_rgb_camera_optical_frame`) and
    the `map` frame must be available in the TF2 buffer.

How to run:
1. Source your ROS 2 and workspace environment.
2. Ensure the cone detection node is running and publishing on
    `/detected_cones` and the camera publishes `/oakd/stereo/camera_info`.
3. Run the node:
        ros2 run nav_pkg pose_estimator_node
"""

import rclpy
import rclpy.duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import rclpy.time
import tf2_geometry_msgs
from tf2_geometry_msgs import PointStamped  # This registers the type with tf2
import math
import threading
from nav_interfaces.msg import ConeDetection, ConeDetectionArray
import tf2_ros
from sensor_msgs.msg import CameraInfo

class PoseEstimatorNode(Node):
    """
    ROS2 node that computes an intermediate waypoint from cone detections.

    Responsibilities
    - Listen to cone detections and camera intrinsics.
    - Convert pixel+depth information into 3D points in the camera frame.
    - Transform points into the `map` frame using TF2.
    - Compute either the midpoint between two aligned cones or a lateral
      offset point for a single cone.
    - Publish the resulting `PoseStamped` to `/intermediate_waypoint`.

    Attributes
        bot_width (float): Robot footprint width (m) used when computing lateral offsets.
        add_space (float): Extra lateral clearance to add when computing offset.
        _pose_lock (threading.Lock): Protects access to `current_robot_pose`.
        camera_ready (bool): True once camera intrinsics have been received.
        current_robot_pose (geometry_msgs.msg.Pose): Latest AMCL pose (thread-safe).
        rgb_frame_id (str): Camera optical frame used for TF lookups.
        tf_buffer (tf2_ros.Buffer): TF2 buffer used to lookup transforms.
        tf_listener (tf2_ros.TransformListener): TF2 listener attached to the node.
        cone_sub (rclpy.subscription.Subscription): Subscription for `/detected_cones`.
        publisher (rclpy.publisher.Publisher): Publisher for `/intermediate_waypoint`.

    Behavioural notes
        - The node uses conservative timeouts when waiting for TF transforms
          and will skip detection callbacks until both camera info and TF
          lookups succeed.
        - Many numeric values (offsets, depth thresholds) are currently
          hard-coded and should be converted to ROS parameters for
          production use.
    """

    def __init__(self):
        super().__init__('pose_estimator_node')

        self.bot_width = 0.20
        self.add_space = 0.3
        self._pose_lock = threading.Lock()

        self.camera_ready = False
        self.model = None
        self.current_robot_pose = None

        self.depth_width = 1280
        self.depth_height = 720
        self.rgb_frame_id = "oakd_rgb_camera_optical_frame"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.cone_sub = self.create_subscription(ConeDetectionArray, '/detected_cones', self.cone_callback, 10)
        # Camera info
        self._camera_info_sub = self.create_subscription(CameraInfo, '/oakd/stereo/camera_info', self._camera_info_callback, 10)

        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_callback, 10)

        self.publisher = self.create_publisher(PoseStamped, '/intermediate_waypoint', 10)

    def _camera_info_callback(self, msg):
        """
        Callback for `/oakd/stereo/camera_info`.

        Stores camera intrinsic parameters (fx, fy, cx, cy) on first
        reception and marks the camera as ready. The subscription is
        destroyed after the first valid message since intrinsics are
        assumed static for the runtime of the node.

        Args:
            msg (sensor_msgs.msg.CameraInfo): Camera intrinsics message.
        """
        if not self.camera_ready:
            self.camera_info = msg
            # CameraInfo.k follows row-major intrinsic matrix ordering
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_ready = True
            # No longer need intrinsics subscription once read
            self.destroy_subscription(self._camera_info_sub)
    
    def _amcl_pose_callback(self, msg):
        """
        Callback for `/amcl_pose`.

        Updates the node's cached robot pose in a thread-safe manner. The
        stored pose can be used by future heuristics (e.g. to compute a
        goal orientation) though currently only stored for completeness.

        Args:
            msg (geometry_msgs.msg.PoseWithCovarianceStamped): AMCL pose
                estimate.
        """
        # Thread-safe pose update
        with self._pose_lock:
            self.current_robot_pose = msg.pose.pose

    def publish_mid_pose(self, mid_pose=None):
        """
        Publish a computed intermediate waypoint.

        Builds a `PoseStamped` in the `map` frame from the provided (x,y)
        tuple and publishes it on `/intermediate_waypoint`.

        Args:
            mid_pose (tuple[float,float] | None): (x, y) coordinates in map
                frame to publish. If None the function does nothing.
        """

        x, y = mid_pose
        new_pose = self.getPoseStamped([x, y], 0)
        self.publisher.publish(new_pose)
        return

    def getPoseStamped(self, position, rotation):
        """
        Fill and return a PoseStamped message.

        :param position: A list consisting of the x and y positions for the Pose. e.g [0.5, 1.2]
        :param rotation: Rotation of the pose about the Z axis in degrees.
        :return: PoseStamped message
        """
        pose = PoseStamped()

        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]

        # Convert Z rotation to quaternion
        pose.pose.orientation.z = math.sin(math.radians(rotation) / 2)
        pose.pose.orientation.w = math.cos(math.radians(rotation) / 2)

        return pose

    def cone_callback(self, msg):
        """
        Main callback for `/detected_cones`.

        Behaviour:
        - Waits until camera intrinsics are available.
        - Chooses the closest cone from the detection array.
        - Filters out cones that are too close or too far using simple
          distance thresholds.
        - If an aligned cone of the opposite colour is detected, computes
          the midpoint between the two cones in the map frame.
        - Otherwise computes a single-cone lateral offset point.
        - Publishes the resulting intermediate waypoint if available.

        Args:
            msg (nav_interfaces.msg.ConeDetectionArray): Array of detected cones.
        """
        if not self.camera_ready:
            self.get_logger().warn("Camera not yet ready. Skipping cone callback.")
            return
        
        closest_cone = min(msg.detections, key=lambda c: c.depth)

        color = closest_cone.color
        x1 = closest_cone.x_min
        x2 = closest_cone.x_max
        y1 = closest_cone.y_min
        y2 = closest_cone.y_max
        depth = closest_cone.depth

        # Basic depth gating to avoid spurious near/far detections
        if depth < 1.2:
            self.get_logger().warn("Cone depth is too close")
            return
        if depth > 3.5:
            self.get_logger().warn("Cone depth is too away")
            return

        cx_closest = (x1 + x2) // 2
        cy_closest = (y1 + y2) // 2
        opposite_color = "yellow" if "red" in color else "red"

        aligned = [
            c for c in msg.detections
            if opposite_color in c.color and
            abs(((c.y_min + c.y_max) // 2) - (cy_closest)) < 30
        ]

        cone_pose = None

        if aligned:
            # Use the closest partner among aligned cones
            partner = min(aligned, key=lambda c: c.depth)
            cx_partner = (partner.x_min + partner.x_max) // 2
            cy_partner = (partner.y_min + partner.y_max) // 2

            cone_pose = self.two_cones_transform(depth, (cx_closest, cy_closest), partner.depth, (cx_partner, cy_partner))
            
        else:
            cone_pose = self.single_cone_transform(depth, (cx_closest, cy_closest), color)

        if cone_pose is not None:
            self.publish_mid_pose(cone_pose)

        return
    
    def pixel_to_3d(self, x_rgb, y_rgb, depth_m):
        """
        Convert an RGB pixel and depth into a 3D point in the camera frame.

        The conversion uses the pinhole camera model with stored intrinsics
        (fx, fy, cx, cy). A small heuristic reduces the reported depth at
        larger distances to compensate for systematic bias in detection.

        Args:
            x_rgb (int): Pixel x coordinate in the RGB image.
            y_rgb (int): Pixel y coordinate in the RGB image.
            depth_m (float): Depth value in metres reported by the detector.

        Returns:
            tuple(float, float, float) | None: (X, Y, Z) in camera frame or
            None if intrinsics are not available.
        """
        if self.camera_info is None:
            self.get_logger().warn("Camera info not yet received.")
            return None

        # Heuristic depth adjustment for far cones
        if depth_m > 1.5:
            X = (x_rgb - self.cx) * (depth_m - 0.45) / self.fx
            Y = (y_rgb - self.cy) * depth_m / self.fy
            depth_m -= 0.1
        else:
            X = (x_rgb - self.cx) * depth_m / self.fx
            Y = (y_rgb - self.cy) * depth_m / self.fy

        Z = depth_m

        return X, Y, Z
    
    def two_cones_transform(self, depth_m1, pixel1, depth_m2, pixel2):
        """
        Compute midpoint between two cones and return a map-frame (x,y).

        Procedure:
        - Ensure a TF transform from the camera frame to `map` is available.
        - Convert each cone pixel+depth to a camera-frame 3D point using
          `pixel_to_3d`.
        - Transform both points into the `map` frame using TF2 and compute
          the midpoint.

        Args:
            depth_m1 (float): Depth (m) for the first cone.
            pixel1 (tuple[int,int]): (cx,cy) pixel for the first cone.
            depth_m2 (float): Depth (m) for the second cone.
            pixel2 (tuple[int,int]): (cx,cy) pixel for the second cone.

        Returns:
            tuple(float,float) | None: (x_map, y_map) midpoint or None on error.
        """
        try:
            # Timeout is short to keep callback responsive
            if not self.tf_buffer.can_transform(
                "map", self.rgb_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            ):
                self.get_logger().warn("Transform not available yet")
                return None

            transform = self.tf_buffer.lookup_transform(
                "map", self.rgb_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        def to_map_point(u, v, depth_m):
            cam_point = self.pixel_to_3d(u, v, depth_m)
            if cam_point is None:
                return None
            x_cam, y_cam, z_cam = cam_point

            point = PointStamped()
            point.header.frame_id = self.rgb_frame_id
            point.header.stamp = rclpy.time.Time().to_msg()
            point.point.x = x_cam
            point.point.y = y_cam
            point.point.z = z_cam

            point_map = tf2_geometry_msgs.do_transform_point(point, transform)
            self.get_logger().info(
                f"Map base : ({point_map.point.x}, {point_map.point.y})"
            )
            return point_map

        # Transform the two cones to map frame
        point1 = to_map_point(pixel1[0], pixel1[1], depth_m1)
        point2 = to_map_point(pixel2[0], pixel2[1], depth_m2)

        x_goal = round((point1.point.x + point2.point.x) / 2, 2)
        y_goal = round((point1.point.y + point2.point.y) / 2, 2)

        return x_goal, y_goal

    def single_cone_transform(self, depth_m, pixel, color):
        """
        Compute a lateral intermediate point for a single cone.

        Procedure:
        - Ensure TF from camera to `map` is available.
        - Convert the cone pixel+depth to a 3D camera-frame point.
        - Apply a lateral offset (depends on cone colour) to produce a
          waypoint that safely clears the cone.
        - Transform the offset point to the `map` frame and return it.

        Args:
            depth_m (float): Depth (m) for the cone.
            pixel (tuple[int,int]): (cx,cy) pixel for the cone.
            color (str): Detected color label (used to choose side).

        Returns:
            tuple(float,float) | None: (x_map, y_map) or None on error.
        """
        try:
            if not self.tf_buffer.can_transform(
                "map", self.rgb_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            ):
                self.get_logger().warn("Transform not available yet")
                return None

            transform_map = self.tf_buffer.lookup_transform(
                "map", self.rgb_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        # === STEP 1: Compute camera-frame 3D point ===
        cam_point = self.pixel_to_3d(pixel[0], pixel[1], depth_m)

        x, y, z = cam_point
        self.get_logger().info(f"Original camera point: ({x:.2f}, {y:.2f}, {z:.2f})")

        # === STEP 2: Apply lateral offset ===
        side = +1 if "red" in color.lower() else -1
        offset_side = side * 0.7
        x += offset_side
        self.get_logger().info(f"Offset applied in camera frame: {offset_side:.2f} → new x: {x:.2f}")

        # === STEP 3: Transform to map frame ===
        point = PointStamped()
        point.header.frame_id = self.rgb_frame_id
        point.header.stamp = rclpy.time.Time().to_msg()
        point.point.x = x
        point.point.y = y
        point.point.z = z

        point_map = tf2_geometry_msgs.do_transform_point(point, transform_map)
        self.get_logger().info(f"Final map point: ({point_map.point.x:.2f}, {point_map.point.y:.2f})")

        x_goal = round(point_map.point.x, 2)
        y_goal = round(point_map.point.y, 2)
        return x_goal, y_goal

def main(args=None):
    rclpy.init(args=args)

    node = PoseEstimatorNode()

    try:
        rclpy.spin(node) #Running loop - bocking call
    except KeyboardInterrupt:
        rclpy.shutdown()
        pass
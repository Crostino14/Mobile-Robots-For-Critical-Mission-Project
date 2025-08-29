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

""" Tutte le variabili, metodi e invocazioni commentati servono per provare a calcolare il punto intermedio laterale ai coni rispetto alla mappa piuttosto che rispetto al camera frame, perché ho notato che i punti ad esempio vengono generati avanti al cono nelle direzione del goal finale, piuttosto che effettivamente a lato. È solo un'idea e va testata a settembre, non è stata proprio provata"""

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')

        self.bot_width = 0.20
        self.add_space = 0.3
        self._pose_lock = threading.Lock()

        self.camera_ready = False
        self.model = None
        self.current_robot_pose = None

        self.depth_width = 1280  # Imposta questi in base alla tua pipeline
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
        if not self.camera_ready:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4] 
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_ready = True
            self.destroy_subscription(self._camera_info_sub)
    
    def _amcl_pose_callback(self, msg):
        #Thread-safe pose update
        with self._pose_lock:
            self.current_robot_pose = msg.pose.pose

    def publish_mid_pose(self, mid_pose=None):

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
            print("\nConi allineati")
            # 4. Calcola punto medio (pixel o spaziale)
            partner = min(aligned, key=lambda c: c.depth)  # opzionale

            cx_partner = (partner.x_min + partner.x_max) // 2

            cy_partner = (partner.y_min + partner.y_max) // 2

            cone_pose = self.two_cones_transform(depth, (cx_closest, cy_closest), partner.depth, (cx_partner, cy_partner))
            
        else:
            print(f"\nCono singolo {color} {cx_closest} {cy_closest}")
            cone_pose = self.single_cone_transform(depth, (cx_closest, cy_closest), color)
        
        if cone_pose is not None:

            self.publish_mid_pose(cone_pose)

        return
    
    def pixel_to_3d(self, x_rgb, y_rgb, depth_m):
        if self.camera_info is None:
            self.get_logger().warn("Camera info not yet received.")
            return None

        if depth_m > 1.5:
            X = (x_rgb - self.cx) * (depth_m - 0.45) / self.fx
            Y = (y_rgb - self.cy) * (depth_m) / self.fy
            depth_m -= 0.1
        else:
            X = (x_rgb - self.cx) * (depth_m ) / self.fx
            Y = (y_rgb - self.cy) * (depth_m) / self.fy
        
        Z = depth_m

        return X, Y, Z
    
    def two_cones_transform(self, depth_m1, pixel1, depth_m2, pixel2):

        try:
            
            if not self.tf_buffer.can_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds = 1.0)):
                self.get_logger().warn("Transform not available yet")
                return None
            
            transform = self.tf_buffer.lookup_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds = 1.0))

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        def to_map_point(u, v, depth_m):
            x_cam, y_cam, z_cam = self.pixel_to_3d(u, v, depth_m)

            point = PointStamped()
            point.header.frame_id = self.rgb_frame_id
            point.header.stamp = rclpy.time.Time().to_msg()
            point.point.x = x_cam
            point.point.y = y_cam
            point.point.z = z_cam

            point_map = tf2_geometry_msgs.do_transform_point(point, transform)

            self.get_logger().info(f"Map base : ({point_map.point.x}, {point_map.point.y})")
            return point_map

        # Trasforma i due coni
        point1 = to_map_point(pixel1[0], pixel1[1], depth_m1)
        point2 = to_map_point(pixel2[0], pixel2[1], depth_m2)

        x_goal = round((point1.point.x + point2.point.x) / 2, 2)
        y_goal = round((point1.point.y + point2.point.y) / 2, 2)

        return x_goal, y_goal

    def single_cone_transform(self, depth_m, pixel, color):
        # 1) Trasformazioni necessarie: camera->base_link e base_link->map
        try:
            if not self.tf_buffer.can_transform("base_link", self.rgb_frame_id, rclpy.time.Time(),
                                                timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn("Transform camera->base_link non pronto")
                return None
            T_cam_to_bl = self.tf_buffer.lookup_transform(
                "base_link", self.rgb_frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            if not self.tf_buffer.can_transform("map", "base_link", rclpy.time.Time(),
                                                timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn("Transform base_link->map non pronto")
                return None
            T_bl_to_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        # 2) Punto 3D del cono nel frame camera
        u, v = pixel
        x_cam, y_cam, z_cam = self.pixel_to_3d(u, v, depth_m)
        p_cam = PointStamped()
        p_cam.header.frame_id = self.rgb_frame_id
        p_cam.header.stamp = rclpy.time.Time().to_msg()
        p_cam.point.x, p_cam.point.y, p_cam.point.z = x_cam, y_cam, z_cam

        # 3) Porta il cono in base_link
        p_bl = tf2_geometry_msgs.do_transform_point(p_cam, T_cam_to_bl)
        x_bl, y_bl = p_bl.point.x, p_bl.point.y
        self.get_logger().info(f"Cone in base_link: ({x_bl:.2f}, {y_bl:.2f})")

        # 4) Crea il gemello virtuale in base_link spostando SOLO lungo y
        #    half_gate = metà larghezza porta desiderata (robot/2 + margine)
        half_gate = 0.8  # es. 0.1 + 0.3 = 0.4 m
        #    Il gemello sta a ±(2*half_gate) in y; il centro sarà a ±half_gate.
        color_l = color.lower()
        if "yellow" in color_l:
            # devo passare a sinistra del giallo → rosso virtuale a destra (y - 2*half_gate)
            y_center_bl = y_bl + half_gate
        elif "red" in color_l:
            # devo passare a destra del rosso → giallo virtuale a sinistra (y + 2*half_gate)
            y_center_bl = y_bl - half_gate
        else:
            self.get_logger().warn("Colore cono non riconosciuto, abort.")
            return None

        x_center_bl = x_bl  # stessa x del cono (nessuna correzione angolare, come richiesto)

        # 5) Trasforma il centro porta in map (equivalente alla media dopo trasformazione rigida)
        center_bl = PointStamped()
        center_bl.header.frame_id = "base_link"
        center_bl.header.stamp = rclpy.time.Time().to_msg()
        center_bl.point.x = x_center_bl
        center_bl.point.y = y_center_bl
        center_bl.point.z = 0.0

        center_map = tf2_geometry_msgs.do_transform_point(center_bl, T_bl_to_map)

        x_goal = round(center_map.point.x, 2)
        y_goal = round(center_map.point.y, 2)

        self.get_logger().info(f"Single-cone center -> map: ({x_goal:.2f}, {y_goal:.2f})")
        return x_goal, y_goal

def main(args=None):
    rclpy.init(args=args)

    node = PoseEstimatorNode()

    try:
        rclpy.spin(node) #Running loop - bocking call
    except KeyboardInterrupt:
        rclpy.shutdown()
        pass
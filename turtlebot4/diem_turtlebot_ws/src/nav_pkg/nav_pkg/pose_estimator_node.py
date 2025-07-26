import rclpy
import rclpy.duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import rclpy.time
import tf2_geometry_msgs
from tf2_geometry_msgs import PointStamped  # This registers the type with tf2
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import json
import math
import argparse
import threading
from nav_interfaces.msg import ConeDetection, ConeDetectionArray
import tf2_ros
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel

""" Tutte le variabili, metodi e invocazioni commentati servono per provare a calcolare il punto intermedio laterale ai coni rispetto alla mappa piuttosto che rispetto al camera frame, perché ho notato che i punti ad esempio vengono generati avanti al cono nelle direzione del goal finale, piuttosto che effettivamente a lato. È solo un'idea e va testata a settembre, non è stata proprio provata"""

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('pose_estimator_node')

        self.bot_width = 0.20
        self.add_space = 0.3
        #self._pose_lock = threading.Lock()

        self.camera_ready = False
        self.model = None
        #self.current_robot_pose = None

        self.navigator = TurtleBot4Navigator()

        self.depth_width = 1280  # Imposta questi in base alla tua pipeline
        self.depth_height = 720
        self.rgb_frame_id = "oakd_rgb_camera_optical_frame"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.cone_sub = self.create_subscription(ConeDetectionArray, '/detected_cones', self.cone_callback, 10)
        # Camera info
        self._camera_info_sub = self.create_subscription(CameraInfo, '/oakd/stereo/camera_info', self._camera_info_callback, 10)

        #self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_pose_callback, 10)

        self.publisher = self.create_publisher(PoseStamped, '/intermediate_waypoint', 10)

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()

    def _camera_info_callback(self, msg):
        if not self.camera_ready:
            self.camera_info = msg
            self.fx = msg.k[0]
            self.fy = msg.k[4] 
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_ready = True
            self.destroy_subscription(self._camera_info_sub)
    
    """def _amcl_pose_callback(self, msg):
        #Thread-safe pose update
        with self._pose_lock:
            self.current_robot_pose = msg.pose.pose"""

    def publish_mid_pose(self, mid_pose=None):

        x, y = mid_pose

        new_pose = self.navigator.getPoseStamped([x, y], 0)

        self.publisher.publish(new_pose)
        return
       
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
        
        # PER SETTEMBRE: soglia inferiore testata, abbassando il robot andava troppo oltre i coni, questa era la migliore

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

        # PER SETTEMBRE: la modifica alla depth era per evitare che punti venissero messi nel od oltre il muro, inizialmente vi era anche nella Y ma provando non abbiamo notato differenze sostanziali.

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
        try:
            if not self.tf_buffer.can_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn("Transform not available yet")
                return None

            transform_map = self.tf_buffer.lookup_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        # === STEP 1: Calcola punto 3D nel frame della camera ===
        x, y, z = self.pixel_to_3d(pixel[0], pixel[1], depth_m)
        self.get_logger().info(f"Original camera point: ({x:.2f}, {y:.2f}, {z:.2f})")

        # === STEP 2: Applica offset laterale ===
        side = +1 if "red" in color.lower() else -1
        offset_side = side * 0.5  # puoi usare self.bot_width / 2 + add_space se preferisci
        x += offset_side
        self.get_logger().info(f"Offset applied in camera frame: {offset_side:.2f} → new y: {y:.2f}")

        # === STEP 3: Trasforma nel frame map ===
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
    

    """def single_cone_transform(self, depth_m, pixel, color):
        try:
            if not self.tf_buffer.can_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
                self.get_logger().warn("Transform not available yet")
                return None

            transform_map = self.tf_buffer.lookup_transform("map", self.rgb_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None

        # === STEP 1: Calcola punto 3D nel frame della camera ===
        x, y, z = self.pixel_to_3d(pixel[0], pixel[1], depth_m)
        self.get_logger().info(f"Original camera point: ({x:.2f}, {y:.2f}, {z:.2f})")

        # === STEP 3: Trasforma nel frame map ===
        point = PointStamped()
        point.header.frame_id = self.rgb_frame_id
        point.header.stamp = rclpy.time.Time().to_msg()
        point.point.x = x
        point.point.y = y
        point.point.z = z

        point_map = tf2_geometry_msgs.do_transform_point(point, transform_map)
        self.get_logger().info(f"Final map point: ({point_map.point.x:.2f}, {point_map.point.y:.2f})")

        # 2. Estrai yaw del robot
        with self._pose_lock:
            if self.current_robot_pose is None:
                return None
            yaw = self.extract_yaw(self.current_robot_pose)

        # 3. Calcola il versore laterale sinistro rispetto allo yaw
        lateral = [-math.sin(yaw), math.cos(yaw)]  # Versore laterale sinistro

        # 4. Applica l'offset dal cono nella direzione corretta
        if color == 'yellow':
            sign = +1
        elif color == 'red':
            sign = -1
        else:
            self.get_logger().warn("Colore cono non valido, uso offset neutro.")
            sign = 0

        offset_distance = 0.5 

        dx = sign * offset_distance * lateral[0]
        dy = sign * offset_distance * lateral[1]

        x_goal = point_map.point.x + dx
        y_goal = point_map.point.x + dx

        return x_goal, y_goal
    
    def extract_yaw(self, pose_stamped):
        #Estrai yaw da un quaternion.
        q = pose_stamped.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)"""

def main(args=None):
    rclpy.init(args=args)

    node = PoseEstimatorNode()

    try:
        rclpy.spin(node) #Running loop - bocking call
    except KeyboardInterrupt:
        rclpy.shutdown()
        pass
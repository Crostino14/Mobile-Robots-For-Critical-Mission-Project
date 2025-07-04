import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
from nav_interfaces.msg import ConeDetection, ConeDetectionArray

# === Main ROS2 Node ===
class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/oakd/stereo/image_raw',  # o altro topic di profondità reale
            self.depth_callback,
            10)
        
        #self.cone_pub = self.create_publisher(String, '/detected_cones', 10)
        self._last_depth = None
        self.model = YOLO("/home/chiara/Documents/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/src/nav_pkg/nav_pkg/cone.pt")
        self.get_logger().info("YOLOv8 model loaded successfully")

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return
        
        results = self.model(frame, verbose=False)[0]

        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cls_id = int(det.cls[0].item())
            conf = float(det.conf[0].item())

            # Rileva colore dominante
            label, rgb, hue = self.dominant_color([x1, y1, x2, y2], frame)

            cx_rgb = (x1 + x2) // 2  # Coordinate nel frame RGB
            cy_rgb = (y1 + y2) // 2

            # Converti a coordinate depth
            cx_depth = int(cx_rgb * (1200/250))
            cy_depth = int(cy_rgb * (720/250))

            if self._last_depth is not None:
                distance = float(self._last_depth[int(cy_depth), int(cx_depth)]/1000)
            else:
                distance = 0.0
            
            if distance != 0.0 and distance <= 5.0:
                self.get_logger().info(f"Color: {label}, Distance: {distance:.2f}, Center X : {cx_rgb}, Center Y : {cy_rgb}")

                cone_msg = ConeDetection()

                # Popola i campi del messaggio
                cone_msg.x_min = x1
                cone_msg.y_min = y1
                cone_msg.x_max = x2
                cone_msg.y_max = y2
                cone_msg.distance = distance
                cone_msg.color = label
            
            # Etichetta
            cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 2)
            cv2.putText(frame, f'{label} Cone', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)

        cv2.imshow("Traffic Cone Detector GUI", frame)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        if msg.encoding not in ['16UC1', '32FC1']:
            self.get_logger().error(f"Formato depth non supportato: {msg.encoding}")
            return
        
        # Converti i dati ROS in un array numpy
        dtype = np.uint16 if msg.encoding == '16UC1' else np.float32
        depth_data = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        
        if depth_data is not None:
            # Salva i dati per l'uso in image_callback
            self._last_depth = depth_data
        
    
    # === Dominant color function ===
    def dominant_color(self, x, img):
        mid_y = int((x[1] + x[3]) / 2)
        box = img[mid_y:int(x[3]), int(x[0]):int(x[2])]
        data = np.reshape(box, (-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

        dominant = centers[0].astype(np.int32)
        bgr = np.uint8([[dominant]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h = hsv[0, 0, 0]
        colors = {'red': [0,0,255], 'yellow': [0,255,255], 'green': [0,255,0],
                'blue': [255,0,0], 'unknown': [50,50,50]}
        if h < 16:
            color = 'red'
        elif h < 35:
            color = 'yellow'
        elif h < 92:
            color = 'green'
        elif h < 130:
            color = 'blue'
        else:
            color = 'unknown'

        return color, colors[color], h
    
def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
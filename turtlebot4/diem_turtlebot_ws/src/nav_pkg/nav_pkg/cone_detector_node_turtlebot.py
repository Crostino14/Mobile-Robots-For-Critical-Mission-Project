import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import String

# === Dominant color function ===
def dominant_color(x, img):
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

# === Main ROS2 Node ===
class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10)
        
        self.model = YOLO("diem_turtlebot_ws/src/nav_pkg/nav_pkg/cone.pt")
        self.get_logger().info("YOLOv8 model loaded successfully")
        self.cone_pub = self.create_publisher(String, '/detected_cones', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, verbose=False)[0]

        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cls_id = int(det.cls[0].item())
            conf = float(det.conf[0].item())

            # Rileva colore dominante
            label, rgb, hue = dominant_color([x1, y1, x2, y2], frame)

            if label == 'yellow':
                side = 'left'
            elif label == 'red':
                side = 'right'
            else:
                side = 'unknown'

            cone_msg = String()
            cone_msg.data = f"{label},{x1},{y1},{x2},{y2},{side}"
            self.cone_pub.publish(cone_msg)

            # Etichetta
            cv2.rectangle(frame, (x1, y1), (x2, y2), rgb, 2)
            cv2.putText(frame, f'{label} Cone', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)

        cv2.imshow("Traffic Cone Detector GUI", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
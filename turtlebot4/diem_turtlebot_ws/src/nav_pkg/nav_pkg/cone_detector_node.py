import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
from nav_interfaces.msg import ConeDetection, ConeDetectionArray
from scipy.spatial.transform import Rotation

class ConeDetectionNode(Node):
    def __init__(self):
        super().__init__('cone_detection_node')
        self.bridge = CvBridge()
        
        # Detection filtering parameters
        self.min_detection_confidence = 0.5
        self.max_detection_distance = 8.0
        self.depth_roi_size = 5

        self._setup_subscriptions()
        
        self.cone_pub = self.create_publisher(ConeDetectionArray, '/detected_cones', 10)
        
        self.depth_image = None
        self.model = YOLO("/home/ago/Documenti/GitHub/Mobile-Robots-For-Critical-Mission-Project/turtlebot4/diem_turtlebot_ws/src/nav_pkg/nav_pkg/cone.pt")
        self.get_logger().info("YOLOv8 model loaded successfully")
        
        # RGB image dimensions (will be set from the first RGB message)
        self.rgb_width = None
        self.rgb_height = None

    def _setup_subscriptions(self):
        """Setup subscriptions for image and depth data."""

        # Subscribe to RGB image topic
        self.img_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        # Subscribe to depth image topic
        self.depth_sub = self.create_subscription(
            Image,
            '/oakd/stereo/image_raw',
            self.depth_callback,
            10
        )


    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        img_resized = cv2.resize(frame, (1280, 720))
        # Save RGB dimensions for depth processing
        self.rgb_width = 1280
        self.rgb_height = 720

        # === 1. YOLO DETECTION ===
        _, detection_data = self.detect_cones(img_resized)
        
        # === 2. BUILD AND PUBLISH DETECTED CONES ===
        detected_cones = []
        for det in detection_data:
            # Calculate distance directly
            distance = self._get_distance_for_detection(det)
            if distance is None or distance <= 0:
                continue  # Skip cones without valid distance
            
            # Skip cones that are too far away
            if distance > self.max_detection_distance:
                continue

            color = det['color']
            cx, cy = det['center']
            
            # Build the message
            x1, y1, x2, y2 = det['bbox']
            cone_msg = ConeDetection()
            cone_msg.color = color
            cone_msg.x_min, cone_msg.y_min = int(x1), int(y1)
            cone_msg.x_max, cone_msg.y_max = int(x2), int(y2)
            cone_msg.cx, cone_msg.cy = int(cx), int(cy)
            cone_msg.depth = float(distance)
            detected_cones.append(cone_msg)

        if len(detected_cones) > 0:
            self.publish_cones(detected_cones)
        
        # === 3. VISUALIZATION ===
        self._visualize_detections(img_resized, detected_cones)

    # PER SETTEMBRE: qui potremmo provare a fare come charlotte e usare una dimensione verticale del patch minore rispetto a quella orizzontale, per evitare di prendere troppo pavimento.
    def _get_distance_for_detection(self, detection_data):
        """Estrae la distanza media da una regione del depth map, scalando le coordinate RGB su stereo."""
        if self.depth_image is None:
            return None

        # Estrai il centroide della detection (coordinate RGB)
        center = detection_data['center']
        cx_rgb, cy_rgb = center

        # Ottieni dimensioni immagini
        depth_height = self.depth_image.shape[0]  # 720
        depth_width = self.depth_image.shape[1]   # 1280

        if self.rgb_width is None or self.rgb_height is None:
            self.get_logger().warn("Dimensioni immagine RGB non disponibili, impossibile mappare i pixel.")
            return None
        
        local_depth = self.depth_image.copy()

        # Definisci una piccola regione attorno al centroide stereo
        y_min = max(0, cy_rgb - self.depth_roi_size)
        y_max = min(depth_height, cy_rgb + self.depth_roi_size)
        x_min = max(0, cx_rgb - self.depth_roi_size)
        x_max = min(depth_width, cx_rgb + self.depth_roi_size)

        # Estrai la regione di interesse (ROI) dal depth map
        depth_roi = local_depth[y_min:y_max, x_min:x_max]

        # Calcola la distanza media nella ROI, escludendo i valori zero (assenza di profondità)
        valid_depths = depth_roi[depth_roi > 0]
        if valid_depths.size == 0:
            return None  # Nessun valore di profondità valido nella ROI
        # Restituisci la distanza media (in mm, quindi converti in metri)
        return np.median(valid_depths) / 1000.0  # mm → m

    def _visualize_detections(self, frame, detected_cones):
        """
        Visualizza i coni rilevati sul frame.
        """
        for cone in detected_cones:
            x1, y1, x2, y2 = cone.x_min, cone.y_min, cone.x_max, cone.y_max
            color_name = cone.color
            distance = cone.depth
            
            if color_name == 'red':
                box_color = (0, 0, 255)
            elif color_name == 'yellow':
                box_color = (0, 255, 255)
            else:
                box_color = (128, 128, 128)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            label = f"{color_name} {distance:.2f}m"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        cv2.imshow("Cone Detections", frame)
        cv2.waitKey(1)

    def detect_cones(self, frame):
        """
        Preprocess the frame for low-light conditions and run YOLO detection.
        Returns the detections, detection data, and the enhanced frame.
        """

        # Run YOLO detection on the enhanced frame
        results = self.model(frame, verbose=False)[0]

        detections = []
        detection_data = []
        
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item())
            
            # Skip detections too close to top
            if y1 < 50 or y2 > 710:
                continue

            # 3. Color analysis with anti-reflection
            cx_rgb = (x1 + x2) // 2
            cy_rgb = (y1 + y2) // 2
            label, rgb, hue = self.dominant_color([x1, y1, x2, y2], frame)

            # Skip unknown colors
            if label not in ['yellow', 'red']:
                self.get_logger().debug(f"Skipping detection with color: {label}")
                continue

            # 4. Additional quality checks
            # Check bbox size and aspect ratio
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            if width < 10 or height < 10:  # Too small
                continue

            if aspect_ratio > 2.5 or aspect_ratio < 0.3:  # Invalid aspect ratio
                continue

            # 5. Confidence adjustment based on position (removed edge and floor filters)
            adjusted_conf = conf

            # Final confidence check
            if adjusted_conf < self.min_detection_confidence:
                continue

            detection_data.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx_rgb, y2),
                'color': label,
                'conf': adjusted_conf,
                'original_conf': conf
            })
            detections.append(([x1, y1, x2-x1, y2-y1], adjusted_conf, label))

        self.get_logger().debug(f"Filtered {len(results.boxes)} raw detections to {len(detection_data)} valid detections")
        return detections, detection_data
    
    def publish_cones(self, detected_cones):
        """
        Publishes the detected cones as a ConeDetectionArray message.
        """
        cone_array_msg = ConeDetectionArray()
        cone_array_msg.detections = detected_cones
        self.cone_pub.publish(cone_array_msg)

    def depth_callback(self, msg):
        """
        Callback per il topic delle immagini di profondità.
        Gestisce la conversione dei dati e mantiene la risoluzione originale.
        """
        if msg.encoding not in ['16UC1', '32FC1']:
            self.get_logger().error(f"Formato depth non supportato: {msg.encoding}")
            return
        # Converti i dati ROS in un array numpy (mantieni originale 1280x720)
        dtype = np.uint16 if msg.encoding == '16UC1' else np.float32
        depth_data_original = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        # Salva i dati originali per l'uso in image_callback
        self.depth_image = depth_data_original

    def dominant_color(self, x, img):
        """
        Determines the dominant color within a bounding box with advanced color boost preprocessing.
        Applies selective color enhancement to improve detection accuracy in variable lighting.
        """
        x1, y1, x2, y2 = x
        
        # Use middle portion of bbox to avoid edge artifacts
        mid_y_start = int(y1 + (y2 - y1) * 0.3)
        mid_y_end = int(y1 + (y2 - y1) * 0.7)
        mid_x_start = int(x1 + (x2 - x1) * 0.2)
        mid_x_end = int(x1 + (x2 - x1) * 0.8)
        
        # Ensure valid crop region
        mid_y_start = max(y1, mid_y_start)
        mid_y_end = min(y2, mid_y_end)
        mid_x_start = max(x1, mid_x_start)
        mid_x_end = min(x2, mid_x_end)
        
        if mid_y_end <= mid_y_start or mid_x_end <= mid_x_start:
            # Fallback to full bbox if middle region is invalid
            box = img[y1:y2, x1:x2]
        else:
            # Use middle region for more stable color detection
            box = img[mid_y_start:mid_y_end, mid_x_start:mid_x_end]
        
        if box.size == 0:
            return 'unknown', [50, 50, 50], 0
        
        # =============================================
        # ADVANCED COLOR BOOST PREPROCESSING
        # =============================================
        
        # 1. ADAPTIVE COLOR ENHANCEMENT
        enhanced_box = self._apply_adaptive_color_boost(box)
        
        # 2. BACKGROUND SUPPRESSION 
        # Remove background colors that could interfere with cone detection
        filtered_box = self._suppress_background_interference(enhanced_box)
        
        # 3. SELECTIVE COLOR TARGET ENHANCEMENT
        # Boost specific color ranges (red, yellow) that correspond to cones
        target_enhanced_box = self._enhance_target_colors(filtered_box)
        
        # =============================================
        # ROBUST COLOR ANALYSIS
        # =============================================
        
        # Use the enhanced box for color analysis
        analysis_box = target_enhanced_box
        
        # Convert to multiple color spaces for robust analysis
        color_analysis = self._multi_colorspace_analysis(analysis_box)
        
        # K-means clustering on enhanced data
        dominant_info = self._robust_dominant_color_extraction(analysis_box)
        
        # Enhanced color classification with confidence scoring
        color_result = self._enhanced_color_classification(
            dominant_info, color_analysis, analysis_box
        )
        
        return color_result['name'], color_result['bgr'], color_result['hue']

    def map_rgb_pixel_to_stereo_pixel(self, pixel_rgb, depth_image_original):
        """
        Mappa un pixel RGB a un pixel stereo e calcola la profondità usando l'immagine intera.
        RGB: 400x400, Depth originale: 1280x720.
        
        Args:
            pixel_rgb: tupla (cx_rgb, cy_rgb) del pixel RGB
            depth_image_original: immagine depth originale 1280x720
        Returns:
            tupla (u_stereo, v_stereo, depth_value, depth_image_original)
        """
        cx_rgb, cy_rgb = pixel_rgb
        
        if depth_image_original is None:
            return None, None, 0.0, None

        depth_height = depth_image_original.shape[0]  # 720
        depth_width = depth_image_original.shape[1]   # 1280
        
        if self.rgb_width is None or self.rgb_height is None:
            self.get_logger().warn("Dimensioni immagine RGB non disponibili, impossibile mappare i pixel.")
            return None, None, 0.0, depth_image_original

        # Scala le coordinate RGB (400x400) alle coordinate dell'immagine di profondità (1280x720)
        u_stereo = int(cx_rgb * depth_width / self.rgb_width)
        v_stereo = int(cy_rgb * depth_height / self.rgb_height)

        distance = 0.0
        # Estrai la profondità dall'immagine originale non ritagliata
        if (0 <= v_stereo < depth_height and 0 <= u_stereo < depth_width):
            distance = depth_image_original[v_stereo, u_stereo]
            
            # Converti da mm a metri se necessario
            if distance > 0:
                distance = float(distance) / 1000.0  # mm -> m
            else:
                distance = 0.0
        else:
            # Le coordinate calcolate sono fuori dai limiti dell'immagine di profondità
            distance = 0.0
        
        # Log di debug
        self.get_logger().debug(f"RGB({cx_rgb}, {cy_rgb}) -> Stereo({u_stereo}, {v_stereo}), depth={distance:.3f}m")
        
        # Restituisce l'immagine originale invece di quella croppata
        return u_stereo, v_stereo, distance, depth_image_original

    def _apply_adaptive_color_boost(self, box):
        """
        Apply adaptive color boost to enhance cone colors based on lighting conditions.
        """
        if box.size == 0:
            return box
        
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Analyze lighting conditions
        mean_brightness = np.mean(v)
        
        # Adaptive boost based on brightness
        if mean_brightness < 80:  # Dark conditions
            # Boost saturation more aggressively
            s_boost = 1.6
            v_boost = 1.3
        elif mean_brightness > 180:  # Bright conditions
            # Gentle boost to avoid oversaturation
            s_boost = 1.2
            v_boost = 1.1
        else:  # Normal conditions
            s_boost = 1.4
            v_boost = 1.2
        
        # Apply boost to saturation
        s_float = s.astype(np.float32)
        s_float = np.clip(s_float * s_boost, 0, 255)
        s = s_float.astype(np.uint8)
        
        # Apply boost to value (brightness)
        v_float = v.astype(np.float32)
        v_float = np.clip(v_float * v_boost, 0, 255)
        v = v_float.astype(np.uint8)
        
        # Merge back and convert to BGR
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_box = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced_box

    def _suppress_background_interference(self, box):
        """
        Suppress background colors that might interfere with cone detection.
        """
        if box.size == 0:
            return box
        
        # Convert to HSV
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Define background color ranges to suppress (in HSV)
        # Green vegetation: H=40-80, suppress if low saturation
        green_mask = (h >= 40) & (h <= 80) & (s < 80)
        
        # Blue sky: H=100-130, suppress if high value
        blue_mask = (h >= 100) & (h <= 130) & (v > 150)
        
        # Gray/white surfaces: low saturation across all hues
        gray_mask = (s < 30) & (v > 100)
        
        # Brown/dirt: H=10-25, low saturation
        brown_mask = (h >= 10) & (h <= 25) & (s < 60)
        
        # Combine all background masks
        background_mask = green_mask | blue_mask | gray_mask | brown_mask
        
        # Reduce saturation and value for background pixels
        s_suppressed = s.copy().astype(np.float32)
        v_suppressed = v.copy().astype(np.float32)
        
        s_suppressed[background_mask] *= 0.7  # Reduce saturation
        v_suppressed[background_mask] *= 0.8  # Reduce brightness
        
        # Clip and convert back
        s_suppressed = np.clip(s_suppressed, 0, 255).astype(np.uint8)
        v_suppressed = np.clip(v_suppressed, 0, 255).astype(np.uint8)
        
        # Merge and convert back to BGR
        suppressed_hsv = cv2.merge([h, s_suppressed, v_suppressed])
        suppressed_box = cv2.cvtColor(suppressed_hsv, cv2.COLOR_HSV2BGR)
        
        return suppressed_box

    def _enhance_target_colors(self, box):
        """
        Enhance specific color ranges that correspond to cone colors (red, yellow).
        """
        if box.size == 0:
            return box
        
        # Convert to HSV
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Define target color ranges (HSV)
        # Red cones: H=0-10 or H=160-179
        red_mask1 = (h >= 0) & (h <= 10)
        red_mask2 = (h >= 160) & (h <= 179)
        red_mask = red_mask1 | red_mask2
        
        # Yellow cones: H=15-35
        yellow_mask = (h >= 15) & (h <= 35)
        
        # Orange cones: H=5-20
        orange_mask = (h >= 5) & (h <= 20)
        
        # Combine target masks
        target_mask = red_mask | yellow_mask | orange_mask
        
        # Enhanced boost for target colors
        s_enhanced = s.copy().astype(np.float32)
        v_enhanced = v.copy().astype(np.float32)
        
        # Boost saturation and value for target colors
        s_enhanced[target_mask] = np.clip(s_enhanced[target_mask] * 1.8, 0, 255)
        v_enhanced[target_mask] = np.clip(v_enhanced[target_mask] * 1.4, 0, 255)
        
        # Special boost for yellow in low saturation areas
        yellow_low_sat = yellow_mask & (s < 80)
        s_enhanced[yellow_low_sat] = np.clip(s_enhanced[yellow_low_sat] * 2.2, 0, 255)
        
        # Convert back
        s_enhanced = s_enhanced.astype(np.uint8)
        v_enhanced = v_enhanced.astype(np.uint8)
        
        # Merge and convert back to BGR
        enhanced_hsv = cv2.merge([h, s_enhanced, v_enhanced])
        enhanced_box = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return enhanced_box

    def _multi_colorspace_analysis(self, box):
        """
        Analyze color characteristics across multiple color spaces.
        """
        if box.size == 0:
            return {
                'dominant_hue': 0,
                'mean_saturation': 0,
                'hsv_mean': [0, 0, 0],
                'bgr_mean': [0, 0, 0],
                'lab_mean': [0, 0, 0]
            }
        
        # BGR analysis
        bgr_mean = np.mean(box, axis=(0, 1))
        
        # HSV analysis
        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        hsv_mean = np.mean(hsv, axis=(0, 1))
        
        # Find dominant hue
        h_channel = hsv[:, :, 0]
        hist_h = cv2.calcHist([h_channel], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist_h)
        
        # Mean saturation
        mean_saturation = hsv_mean[1]
        
        # LAB analysis
        lab = cv2.cvtColor(box, cv2.COLOR_BGR2LAB)
        lab_mean = np.mean(lab, axis=(0, 1))
        
        return {
            'dominant_hue': int(dominant_hue),
            'mean_saturation': float(mean_saturation),
            'hsv_mean': hsv_mean.tolist(),
            'bgr_mean': bgr_mean.tolist(),
            'lab_mean': lab_mean.tolist()
        }

    def _robust_dominant_color_extraction(self, box):
        """
        Extract dominant color using K-means clustering for robustness.
        """
        if box.size == 0:
            return {
                'dominant_bgr': [0, 0, 0],
                'dominant_hsv': [0, 0, 0],
                'confidence': 0.0
            }
        
        # Reshape for K-means
        pixels = box.reshape(-1, 3).astype(np.float32)
        
        # Apply K-means clustering
        k = 3  # Number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the most frequent cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique_labels[np.argmax(counts)]
        dominant_bgr = centers[dominant_cluster_idx]
        
        # Calculate confidence based on cluster dominance
        confidence = np.max(counts) / len(labels)
        
        # Convert to HSV
        dominant_bgr_uint8 = np.uint8([[dominant_bgr]])
        dominant_hsv = cv2.cvtColor(dominant_bgr_uint8, cv2.COLOR_BGR2HSV)[0, 0]
        
        return {
            'dominant_bgr': dominant_bgr.tolist(),
            'dominant_hsv': dominant_hsv.tolist(),
            'confidence': float(confidence)
        }

    def _enhanced_color_classification(self, dominant_info, color_analysis, analysis_box):
        """
        Enhanced color classification with confidence scoring.
        """
        hsv = dominant_info['dominant_hsv']
        hue = hsv[0]
        saturation = hsv[1]
        value = hsv[2]
        confidence = dominant_info['confidence']
        
        # Color classification based on HSV ranges
        if (hue <= 10 or hue >= 160) and saturation >= 50:
            # Red range
            color_name = 'red'
            base_confidence = 0.8
        elif 15 <= hue <= 35 and saturation >= 30:
            # Yellow range
            color_name = 'yellow'
            base_confidence = 0.7
        elif 5 <= hue <= 20 and saturation >= 40:
            # Orange range (classify as red for cones)
            color_name = 'red'
            base_confidence = 0.6
        elif 40 <= hue <= 80 and saturation >= 50:
            # Green range
            color_name = 'green'
            base_confidence = 0.5
        elif 90 <= hue <= 130 and saturation >= 50:
            # Blue range
            color_name = 'blue'
            base_confidence = 0.5
        else:
            # Unknown/uncertain
            color_name = 'unknown'
            base_confidence = 0.1
        
        # Adjust confidence based on various factors
        final_confidence = base_confidence * confidence
        
        # Penalize low saturation (likely gray/white)
        if saturation < 30:
            final_confidence *= 0.5
        
        # Penalize very low or very high brightness
        if value < 30 or value > 220:
            final_confidence *= 0.7
        
        # Boost confidence for target colors (red, yellow)
        if color_name in ['red', 'yellow']:
            final_confidence *= 1.2
        
        return {
            'name': color_name,
            'bgr': dominant_info['dominant_bgr'],
            'hsv': dominant_info['dominant_hsv'],
            'hue': int(hue),
            'confidence': min(final_confidence, 1.0)
        }

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
Project: Mobile Robots for Critical Mission — nav_pkg.cone_detection_node

Student and Creator:

Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
Chiara Ferraioli         0622702169      c.ferraioli30@studenti.unisa.it
Asja Antonucci           0622702437      a.antonucci5@studenti.unisa.it

Purpose:
This module implements `ConeDetectionNode`, a ROS2 node that performs
cone detection on RGB frames using a YOLOv8 model and extracts depth
information from a synchronised stereo/depth image. The node publishes
`nav_interfaces/ConeDetectionArray` messages for downstream pose estimation
and navigation.

Design summary and responsibilities:
- Subscribe to RGB frames on `/oakd/rgb/preview/image_raw` and depth frames on
    `/oakd/stereo/image_raw`.
- Run a YOLOv8 detector on resized RGB frames to locate cone bounding boxes.
- For each valid detection, compute a robust distance estimate using a small
    depth ROI around the detection centroid.
- Filter detections by confidence, size, aspect ratio and distance.
- Publish filtered detections as `ConeDetectionArray` on `/detected_cones` and
    provide an on-screen visualisation for debugging.

Configuration and assumptions:
- The YOLO model path is currently hard-coded. Prefer a ROS parameter or
    environment variable for portability.
- Depth frames are expected to be full resolution (1280x720) and depth in
    millimetres (16UC1) or metres (32FC1). The node converts mm->m when needed.
- RGB frames are resized to (1280, 720) for consistent mapping to depth.

How to run:
1. Ensure ROS 2 environment is sourced and camera topics are available.
2. Install Python deps (OpenCV, CvBridge, Ultralytics YOLO, SciPy, NumPy).
3. Run the node:
    ros2 run nav_pkg cone_detection_node
"""

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
    """
    ROS2 node that detects traffic cones in RGB images and estimates their depth.

    Responsibilities
    - Subscribe to RGB and depth image topics.
    - Run a YOLOv8 detector to produce bounding boxes and confidences.
    - Analyse the cropped region to classify cone colour (red/yellow) robustly.
    - Estimate distance using a small depth-region median and publish results.

    Attributes
        **bridge (CvBridge)**: Helper to convert ROS Image messages to OpenCV images.
        **min_detection_confidence (float)**: Minimum confidence to accept a detection.
        **max_detection_distance (float)**: Maximum acceptable detection distance (metres).
        **depth_roi_size (int)**: Half-size in pixels of the square ROI used to compute depth.
        **cone_pub (rclpy.publisher.Publisher)**: Publisher for `ConeDetectionArray`.
        **depth_image (np.ndarray|None)**: Last received depth image (original resolution).
        **model (ultralytics.YOLO)**: YOLOv8 model used for detection.
        **rgb_width (int|None)**: Width used for RGB->depth mapping.
        **rgb_height (int|None)**: Height used for RGB->depth mapping.
    """
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
        """Set up subscriptions for RGB and depth image topics.

        This initialises two subscriptions:
        - RGB preview: `/oakd/rgb/preview/image_raw` (Image)
        - Depth image: `/oakd/stereo/image_raw` (Image)

        Each callback is expected to be lightweight; heavy processing occurs in
        `image_callback` which handles detection and publishing.
        """

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
        """
        Process an incoming RGB Image message.

        Workflow:
        1. Convert the ROS Image to an OpenCV BGR image via CvBridge.
        2. Resize the image to the canonical (1280, 720) used by the node.
        3. Run the detector and build `ConeDetection` messages for valid
           detections (colour, bbox, centroid, depth).
        4. Publish valid detections on `/detected_cones` and draw
           a debug visualisation window.

        Args:
            msg (sensor_msgs.msg.Image): Incoming RGB image message.

        Notes:
            - This callback expects the depth image to be available via
              `self.depth_image`; if not, distance estimation will return None
              and the detection will be skipped.
        """
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

    # NOTE: Consider using a smaller vertical patch than horizontal to avoid
    # sampling too much floor in the ROI (idea to be tested).
    def _get_distance_for_detection(self, detection_data):
        """
        Compute a robust distance estimate for a detection.

        The method maps the provided RGB centroid to the depth image space,
        extracts a small square region (defined by `depth_roi_size`) and
        returns the median of the non-zero depth samples converted to metres.

        Args:
            detection_data (dict): A detection dict with at least a 'center'
                entry containing the RGB centroid (cx, cy).

        Returns:
            float | None: Median depth in metres, or None if not computable.

        Behavioural notes:
            - Zero-valued depth pixels are treated as invalid and discarded.
            - If no valid depths remain, the function returns None.
        """
        if self.depth_image is None:
            return None

        # Extract the detection centroid (RGB coordinates)
        center = detection_data['center']
        cx_rgb, cy_rgb = center

        # Obtain depth image dimensions
        depth_height = self.depth_image.shape[0]  # 720
        depth_width = self.depth_image.shape[1]   # 1280

        if self.rgb_width is None or self.rgb_height is None:
            self.get_logger().warn("Dimensioni immagine RGB non disponibili, impossibile mappare i pixel.")
            return None
        
        local_depth = self.depth_image.copy()

        # Define a small square region around the stereo centroid
        y_min = max(0, cy_rgb - self.depth_roi_size)
        y_max = min(depth_height, cy_rgb + self.depth_roi_size)
        x_min = max(0, cx_rgb - self.depth_roi_size)
        x_max = min(depth_width, cx_rgb + self.depth_roi_size)

        # Extract the region of interest (ROI) from the depth map
        depth_roi = local_depth[y_min:y_max, x_min:x_max]

        # Compute the median depth in the ROI, excluding zero values (invalid)
        valid_depths = depth_roi[depth_roi > 0]
        if valid_depths.size == 0:
            return None
        
        return np.median(valid_depths) / 1000.0  # mm → m

    def _visualize_detections(self, frame, detected_cones):
        """
        Draw detected cones on the provided frame and show a debug window.

        This function draws bounding boxes and a small textual label that
        includes the detected colour and the estimated distance in metres.

        Args:
            frame (np.ndarray): BGR image on which to draw.
            detected_cones (list[nav_interfaces.msg.ConeDetection]): List of
                ConeDetection messages to visualise.
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
        Run the YOLOv8 model on the provided frame and post-process raw boxes.

        Steps:
          - Run the model and iterate raw detections.
          - Discard detections too close to image edges or with invalid size/aspect ratio.
          - Compute the dominant colour and filter out non-cone detections.

        Args:
            frame (np.ndarray): BGR image (resized to the node's canonical size).

        Returns:
            tuple: (detections, detection_data) where:
                - detections: list of tuples suitable for quick visualisation
                  [(x, y, w, h), adjusted_conf, label]
                - detection_data: list of dicts with keys:
                    'bbox': [x1, y1, x2, y2]
                    'center': (cx, cy)
                    'color': label
                    'conf': adjusted_conf
                    'original_conf': original_conf
        """

        # Run YOLO detection on the enhanced frame
        results = self.model(frame, verbose=False)[0]

        detections = []
        detection_data = []

        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0].item())

            # Skip detections too close to top or bottom padding
            if y1 < 50 or y2 > 710:
                continue

            # Colour analysis and centroid
            cx_rgb = (x1 + x2) // 2
            cy_rgb = (y1 + y2) // 2
            label, rgb, hue = self.dominant_color([x1, y1, x2, y2], frame)

            # Skip unknown colours
            if label not in ['yellow', 'red']:
                self.get_logger().debug(f"Skipping detection with color: {label}")
                continue

            # Quality checks: size and aspect ratio
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0

            if width < 10 or height < 10:
                continue

            if aspect_ratio > 2.5 or aspect_ratio < 0.3:
                continue

            # Confidence gating
            adjusted_conf = conf
            if adjusted_conf < self.min_detection_confidence:
                continue

            detection_data.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx_rgb, y2),
                'color': label,
                'conf': adjusted_conf,
                'original_conf': conf
            })

            detections.append(([x1, y1, x2 - x1, y2 - y1], adjusted_conf, label))

        self.get_logger().debug(
            f"Filtered {len(results.boxes)} raw detections to {len(detection_data)} valid detections"
        )

        return detections, detection_data
    
    def publish_cones(self, detected_cones):
        """
        Publish an array of detected cones.

        Args:
            detected_cones (list[nav_interfaces.msg.ConeDetection]): A list of
                populated ConeDetection messages.
        """
        cone_array_msg = ConeDetectionArray()
        cone_array_msg.detections = detected_cones
        self.cone_pub.publish(cone_array_msg)

    def depth_callback(self, msg):
        """
        Process an incoming depth Image message and store it for later use.

        This callback expects depth encoding to be either '16UC1' (unsigned
        16-bit millimetres) or '32FC1' (float metres). The image buffer is
        converted to a NumPy array and kept at the native resolution for
        accurate mapping to RGB coordinates.

        Args:
            msg (sensor_msgs.msg.Image): Incoming depth image.
        """
        if msg.encoding not in ['16UC1', '32FC1']:
            self.get_logger().error(f"Unsupported depth encoding: {msg.encoding}")
            return
        # Convert ROS image data to numpy array (preserve original resolution)
        dtype = np.uint16 if msg.encoding == '16UC1' else np.float32
        depth_data_original = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        # Store the original depth buffer for distance estimation
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
        
        # 2. SELECTIVE COLOR TARGET ENHANCEMENT
        # Boost specific color ranges (red, yellow) that correspond to cones
        target_enhanced_box = self._enhance_target_colors(enhanced_box)
        
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
        Map an RGB pixel to stereo (depth) image coordinates and sample depth.

        The function scales the provided RGB pixel coordinates from the node's
        RGB size to the depth image size, then samples the depth buffer at the
        computed location. If the sampled depth is > 0 it is converted to
        metres (mm -> m) for 16-bit encodings; otherwise zero is returned.

        Args:
            pixel_rgb (tuple[int, int]): RGB pixel coordinates (cx, cy).
            depth_image_original (np.ndarray | None): Depth buffer at native resolution.

        Returns:
            tuple: (u_stereo, v_stereo, depth_value_m, depth_image_original)
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
        
        self.get_logger().debug(f"RGB({cx_rgb}, {cy_rgb}) -> Stereo({u_stereo}, {v_stereo}), depth={distance:.3f}m")
        
        # Restituisce l'immagine originale invece di quella croppata
        return u_stereo, v_stereo, distance, depth_image_original

    def _apply_adaptive_color_boost(self, box):
        """
        Apply an adaptive colour boost to a cropped bounding-box region to improve
        cone colour separability in variable lighting.

        Description
        -----------
        This routine converts the provided BGR `box` into HSV space, estimates
        local illumination from the V channel and applies multiplicative gains
        to saturation and value channels. The gains are chosen conservatively to
        avoid clipping while increasing the distinction of target colours
        (red / yellow / orange) typically found on cones.

        Args:
            box (np.ndarray): BGR image patch (H x W x 3). May be empty.

        Returns:
            np.ndarray: Enhanced BGR image patch with the same shape and dtype as
                        the input. If `box` is empty the original array is
                        returned unchanged.

        Notes:
            - Operates on a copy of the input channels; the original `box` is
              not modified in-place by the caller's reference semantics.
            - Uses clipping to keep values in the valid [0, 255] range.
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

    def _enhance_target_colors(self, box):
        """
        Enhance specific colour ranges that correspond to cone colours (red, yellow).

        Description
        -----------
        Performs a selective boost on HSV pixels that fall inside hand-crafted
        hue ranges for red, yellow and orange. The method increases saturation
        and value for target pixels, and applies an extra boost for low-
        saturation yellow regions to recover desaturated cones under poor
        lighting.

        Args:
            box (np.ndarray): BGR image patch (H x W x 3). May be empty.

        Returns:
            np.ndarray: BGR image patch with target colours enhanced. If input
                        is empty the same object is returned.

        Notes:
            - The hue thresholds are deliberately permissive to tolerate
              camera colour shifts; downstream classification applies further
              robustness checks.
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
        Analyse colour characteristics across multiple colour spaces and return
        a compact descriptor useful for later classification.

        Behaviour
        ---------
        The function computes mean colour vectors in BGR, HSV and LAB, builds a
        histogram of hue values to estimate the dominant hue and returns a
        dictionary with the computed statistics.

        Args:
            box (np.ndarray): BGR image patch (H x W x 3). May be empty.

        Returns:
            dict: A dictionary containing:
                - 'dominant_hue' (int): Dominant hue bin [0..179].
                - 'mean_saturation' (float): Mean saturation from HSV.
                - 'hsv_mean' (list): Mean HSV values.
                - 'bgr_mean' (list): Mean BGR values.
                - 'lab_mean' (list): Mean LAB values.

        Notes:
            - If `box` is empty a zero-filled descriptor is returned to avoid
              downstream None checks.
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
        Extract a robust dominant colour using K-means clustering.

        Description
        -----------
        Reshapes the patch pixels into a 2D array and runs K-means (k=3) to
        identify the most frequent cluster. The cluster centre is returned as
        the dominant BGR colour together with a simple confidence score (the
        fraction of pixels assigned to the dominant cluster).

        Args:
            box (np.ndarray): BGR image patch (H x W x 3).

        Returns:
            dict: A dictionary containing:
                - 'dominant_bgr' (list[float]): BGR values of the dominant centre.
                - 'dominant_hsv' (list[int]): HSV of the dominant colour.
                - 'confidence' (float): Cluster dominance ratio in [0,1].

        Notes:
            - Uses a fixed k=3 as a compromise between robustness and speed.
            - Returns zero-valued descriptors for empty inputs.
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
        Enhanced colour classification that converts cluster information and
        multi-space statistics into a categorical label and a confidence score.

        Procedure
        ---------
        - Uses the HSV components of the dominant cluster to locate the colour
          band (red, yellow, orange, etc.).
        - Applies heuristic rules on saturation and value to penalise low-
          confidence cases (e.g. grey/white or overexposed patches).
        - Scales the returned confidence by both cluster dominance and base
          priors for target colours.

        Args:
            dominant_info (dict): Output from `_robust_dominant_color_extraction`.
            color_analysis (dict): Output from `_multi_colorspace_analysis`.
            analysis_box (np.ndarray): The image patch used for analysis.

        Returns:
            dict: A dictionary containing:
                - 'name' (str): One of ['red','yellow','green','blue','unknown'].
                - 'bgr' (list): Dominant BGR colour.
                - 'hsv' (list): Dominant HSV colour.
                - 'hue' (int): Hue channel as integer.
                - 'confidence' (float): Final confidence in [0,1].

        Notes:
            - The classifier is deliberately conservative for non-target colours
              and returns 'unknown' with low confidence when uncertain.
        """
        hsv = dominant_info['dominant_hsv']
        hue = hsv[0]
        saturation = hsv[1]
        value = hsv[2]
        confidence = dominant_info['confidence']

        # Colour classification based on HSV ranges
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

        # Penalise low saturation (likely grey/white)
        if saturation < 30:
            final_confidence *= 0.5

        # Penalise very low or very high brightness
        if value < 30 or value > 220:
            final_confidence *= 0.7

        # Boost confidence for target colours (red, yellow)
        if color_name in ['red', 'yellow']:
            final_confidence *= 1.
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
        # Allow CTRL+C to stop the node gracefully
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
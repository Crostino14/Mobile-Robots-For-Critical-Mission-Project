import rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
from std_msgs.msg import String
from dbr import *
import cv2

class QRCodeReader(Node):
    """
    QRCodeReader is a ROS2 node for reading and decoding QR codes from image frames.

    This node subscribes to an image topic, processes incoming images to detect QR codes,
    and publishes the decoded commands. It uses OpenCV for image processing and a QR code 
    reader library for decoding.

    Attributes:
        command_queue (list): A queue to store detected commands.
        saving_time (float): The time when the last command was saved.
        expiration_time (float): The time threshold for considering a command as expired.
        detection_time (float): The time when a command was detected.

    Methods:
        __init__(): Initializes the QRCodeReader node.
        camera_callback(data): Callback function for processing incoming images.
        publish_command(command): Manages the command detection and saves it in the command queue.
        decodeframe(frame): Decodes a QR code from a given image frame.
    """
    command_queue = []
    saving_time = 0.0
    expiration_time = 15.0
    detection_time = 0.0

    def __init__(self):
        """
        Initialize the QRCodeReader node.

        This constructor initializes the QRCodeReader node by setting up the CvBridge and 
        BarcodeReader instances, initializing the license key for the QR code reader, and 
        creating the ROS2 subscription and publisher.
        """
        super().__init__('qr_code_reader')

        self.bridge = CvBridge()
        self.reader = BarcodeReader()
        #BarCode Reader license key
        license_key = "t0068lQAAADOACSTGONy+TjHVqI+jALoepyU2R3rPZngeVaoMLC2bhT7E2V4tmJYXY/ItSZ4KxD6VctuZYgJAv/mMXeOTXsY=;t0068lQAAAGSCsDCT4KNJsgkYlAzUxqMONjjlFSTh/MQSHzvZnPZFn1lE7bosowW8Obpx+LQKYk3Xn00LQoYzRhe6/tO9ApA="
        self.reader.init_license(license_key)

        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw', 
            self.camera_callback,
            10)
        self.publisher = self.create_publisher(String, 'command', 1)


    def decodeframe(self,frame):
        """
        Decodes a QR code from a given image frame.

        This function attempts to decode a QR code from the provided image frame using the 
        QR code reader. If successful, it returns the decoded text. In case of any errors during 
        decoding, the error is printed.

        :param frame: The image frame containing the QR code to be decoded
        :type frame: numpy.ndarray
        :return: The decoded text from the QR code, or None if no QR code is detected
        :rtype: str or None
        """
        try:
            text_results = self.reader.decode_buffer(frame)
            if text_results:
                return text_results[0].barcode_text
        except BarcodeReaderError as bre:
            print(bre)



    def publish_command(self, command):
        """
        Manages the command detection and saves it in the command queue.

        This function handles the publication of commands decoded from QR codes. It ensures that the same command is not 
        published repeatedly in a short span of time by maintaining a command queue and an expiration time. It updates 
        the queue and the time of the last saved command appropriately based on whether the new command is different or 
        the same as the previous one.

        :param command: The decoded command from the QR code
        :type command: str
        """
        msg = String()
        msg.data = command
        self.detection_time = time.time()

        if not self.command_queue: # queue is empty, save the detected command
            self.command_queue.append(command)
            self.saving_time = self.detection_time
            self.get_logger().info("Publishing signal for the first time")
            self.publisher.publish(msg)
        else:
            if command != self.command_queue[-1]: # new signal is different from the previous one
                self.command_queue.clear() # replace the previous signal with the current one
                self.command_queue.append(command)
                self.saving_time = self.detection_time # update time of signal save
                self.get_logger().info(f"Publishing signal: {msg.data}")
                self.publisher.publish(msg)
            else:
                if (self.detection_time - self.saving_time) >= self.expiration_time: # new signal is equal to the previous one
                    self.saving_time = self.detection_time # just update time of signal save if the new signal has been detected after the time threshold
                    self.get_logger().info(f"Publishing signal: {msg.data}")
                    self.publisher.publish(msg)


    def camera_callback(self, data):
        """
        Callback function for processing incoming images and decoding QR codes.

        This function is triggered whenever a new image message is received from the subscribed topic.
        It converts the ROS Image message to an OpenCV image, attempts to decode any QR codes present
        in the image, and publishes the decoded command.

        :param data: Image message received from the subscribed topic
        :type data: sensor_msgs.msg.Image
        """
        try:
            # Convert the ROS Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return
        
        # Decode the QR code from the image
        command = self.decodeframe(cv_image)
        if command is None:
            command = "NOSIGNAL"
            self.publish_command(command)
        else:
            self.publish_command(command)
        

def main(args=None):
    rclpy.init(args=args)
    qr_code_reader = QRCodeReader()
    rclpy.spin(qr_code_reader)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
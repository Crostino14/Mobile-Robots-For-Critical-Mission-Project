from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
import math

class ConePassagePlanner(Node):
    def __init__(self):
        super().__init__('cone_passage_planner')
        self.navigator = TurtleBot4Navigator()
        self.cones = []
        self.final_goal = None

        self.sub = self.create_subscription(String, '/detected_cones', self.cone_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/final_goal', self.goal_callback, 10)

    def cone_callback(self, msg):
        # msg.data: "yellow,240,100,300,200,left"
        parts = msg.data.split(',')
        if len(parts) != 6:
            return
        color, x1, y1, x2, y2, side = parts
        cone = {
            'color': color,
            'side': side,
            'cx': (int(x1) + int(x2)) // 2,
            'cy': (int(y1) + int(y2)) // 2,
        }
        self.cones.append(cone)
        self.try_plan_between_cones()

    def set_final_goal(self, goal_pose):
        self.get_logger().info("Received final goal pose")
        self.final_goal = goal_pose

    def try_plan_between_cones(self):
        if len(self.cones) < 2:
            return

        # Cerca due coni vicini con lato opposto
        for i in range(len(self.cones)):
            for j in range(i + 1, len(self.cones)):
                a, b = self.cones[i], self.cones[j]
                if a['side'] != b['side']:
                    dist = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])
                    if dist < 100:  # pixels (dipende dalla tua scala)
                        self.get_logger().info(f"Planning between {a['color']} and {b['color']}")
                        mid_x = (a['cx'] + b['cx']) / 2
                        mid_y = (a['cy'] + b['cy']) / 2
                        self.send_mid_goal(mid_x, mid_y)
                        self.cones = []  # svuota dopo invio
                        return

    def send_mid_goal(self, u, v):
        # Camera preview è 250x250; FOV orizzontale ≈ 81°, verticale ≈ 65°
        # Fissiamo una distanza Z ipotetica in metri (quanto sono lontani i coni dalla camera)
        Z = 1.5  # metri: distanza stimata dalla camera ai coni

        # Converti pixel in offset angolari
        cx = 125  # centro immagine
        cy = 125
        hfov = math.radians(81)  # orizzontale
        vfov = math.radians(65)  # verticale

        # Calcola angolo rispetto al centro
        theta_x = (u - cx) / cx * (hfov / 2)
        theta_y = (v - cy) / cy * (vfov / 2)

        # Calcola coordinate in base_link (frame del robot)
        x = Z * math.cos(theta_y) * math.cos(theta_x)
        y = Z * math.cos(theta_y) * math.sin(theta_x)

        # Ora trasformiamo coordinate relative in mappa
        # Supponiamo che il robot sia orientato lungo x positivo e inizialmente posizionato in (0,0)
        # In un caso reale, dovresti trasformare queste coordinate con TF (base_link → map)
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'  # relativa al robot
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        if self.final_goal:
            self.get_logger().info("Re-sending final goal...")
            self.navigator.goToPose(self.final_goal)
        else:
            self.get_logger().info(f"Sending goal: x={x:.2f} m, y={y:.2f} m (from pixel {u:.1f},{v:.1f})")
            self.navigator.goToPose(pose)

def main(args=None):
    rclpy.init(args=args)
    node = ConePassagePlanner()
    rclpy.spin(node)
    rclpy.shutdown()
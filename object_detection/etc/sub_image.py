

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

qos_profile_sensor = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,depth=10,reliability=QoSReliabilityPolicy.BEST_EFFORT,durability=QoSDurabilityPolicy.VOLATILE)


class SubImage(Node):
    def __init__(self):
        super().__init__("sub_image")
        self.image = None
        self.sub_img = self.create_subscription(Image, '/yolo_result', self.img_callback, 10)
        self.bridge = CvBridge()

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data)
        cv2.imshow('frame', self.image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = SubImage()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()    
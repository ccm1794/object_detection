# video1을 subscribe받아서 detection2darray publish
# amz에서 compare box할 때 켜야하는 코드

import sys
sys.path.append('/home/kroad/ros2_ws/src/object_detection/object_detection')
import os
from typing import Tuple, Union, List, Optional
from rclpy.qos import qos_profile_sensor_data, QoSProfile

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
import cv2
import torch
import numpy as np
#import pyrealsense2 as rs
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel

from visualizer import draw_detections
#from ros import create_detection_msg
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose
from std_msgs.msg import Int16
from std_msgs.msg import Float32MultiArray

class ObjectDetection(Node):
    def __init__(self, img_size: Union[Tuple[int, int], None] = (640, 640)
                 ):#class_labels: Union[List, None] = None
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter("weights", "/home/kroad/ros2_ws/src/object_detection/object_detection/bestcone.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("classes_path", "/home/kroad/ros2_ws/src/object_detection/object_detection/bestcone.txt", ParameterDescriptor(description="Class labels file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.25, ParameterDescriptor(description="IOU threshold"))
        #self.declare_parameter("device", "cpu", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))
       
        self.weights = self.get_parameter("weights").get_parameter_value().string_value
        self.classes_path = self.get_parameter("classes_path").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        #self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value

        self.image = None
        self.img_size = img_size
        #self.class_labels
        self.sub_img_flag = False
        
        if self.classes_path:
            if not os.path.isfile(self.classes_path):
                raise FileExistsError(f"classes file not found at {self.classes_path}")
            self.classes = self.parse_classes_file(self.classes_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        if self.classes_path:
            if not os.path.isfile(self.classes_path):
                raise FileExistsError(f"classes file not found at {self.classes_path}")
            self.classes = self.parse_classes_file(self.classes_path)

        # Timer callback
        self.frequency = 50   # Hz : 1초에 20번 타이머 콜백 실행
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for result image
        ccm = QoSProfile(
            reliability = rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            depth=5
            )
        self.visualization_publisher = self.create_publisher(Image, "/yolo_result", 1)
        self.vis_msg = Image()

        # Publishers for detection
        self.detect_publisher = self.create_publisher(Detection2DArray, "/yolo_detect", 1)
        # self.detect_msg = Detection2DArray()

        # Realsense package
        self.bridge = CvBridge()
        # qos_profile_sensor_data
        # Subscribers
        self.sub_img = self.create_subscription(Image, 'video1', self.img_callback, 10)

        self.model = attempt_load(self.weights, map_location=self.device)

    def parse_classes_file(self, path):
        classes = []
        with open(path, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                classes.append(line)
        return classes

    def inference(self, img : torch.Tensor):
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_results = self.model(img)[0]

        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thres, iou_thres = self.iou_thres
        )
        if detections:
            detections = detections[0]
        return detections
    
    def rescale(self, ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
        
        xscale = target_shape[1] / ori_shape[1]
        yscale = target_shape[0] / ori_shape[0]

        boxes[:, [0, 2]] *= xscale
        boxes[:, [1, 3]] *= yscale

        return boxes

    def YOLOv7_detect(self):

        np_img_orig = self.image.copy()

        # 이미지 채널수를 3채널로 바꿔주는 과정
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis = 2)
        
        h_orig, w_orig, c = np_img_orig.shape

        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis = 2)
            c = 3
        
        # 이미지 사이즈를 640*640으로 바꿔준다.
        w_scaled, h_scaled = self.img_size

        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # torch tensor로 변환하는 과정
        img = np_img_resized.transpose((2,0,1))[::-1] #bgr이미지를 rgb순서로 바꿔준다.
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()
        img /= 255 # 이미지에서 텐서로 바꾼 뒤 그 값들을 0~1 사이 값으로 바꿔준다.
        img = img.to(self.device) # 여기까지 모델에 넣어주기 전까지의 과정. 

        detections = self.inference(img)
        detections[:, :4] = self.rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        if len(detections) == 0:
            # print("nothing_detect")
            a=1

        else:
            #### detection2darray로 메세지 보내기 o ####
            box_info = [[float(x1), float(y1), float(x2), float(y2), float(conf), int(c)]
                        for x1, y1, x2, y2, conf, c in detections[:,:].tolist()]
            
            my_bbox = Detection2DArray()

            for bbox_coords in box_info:
                # my_bbox = Detection2DArray()

                # my_bbox.header.frame_id = "fusion_test"
                detection = Detection2D()
                bbox = BoundingBox2D()
                class_id = ObjectHypothesisWithPose()
 
                if bbox_coords[5] == 0: 
                    class_id.id = "blue"
                elif bbox_coords[5] == 1:
                    class_id.id = "orange"

                # detection.score = bbox_coords[4]
                bbox.center.x = float((bbox_coords[0] + bbox_coords[2]) / 2)
                bbox.center.y = float((bbox_coords[1] + bbox_coords[3]) / 2)
                bbox.size_x = float(abs(bbox_coords[2] - bbox_coords[0]))
                bbox.size_y = float(abs(bbox_coords[3] - bbox_coords[1]))

                detection.results.append(class_id)
                detection.bbox = bbox
                my_bbox.detections.append(detection)

            self.detect_publisher.publish(my_bbox)

            
        # if self.visualization_publisher:
        #     bboxes = [[int(x1), int(y1), int(x2), int(y2)]
        #               for x1, y1, x2, y2 in detections[:, :4].tolist()]
        #     classes = [int(c) for c in detections[:, 5].tolist()]
        #     conf = [float(c) for c in detections[:, 4].tolist()]
            
        #     vis_img = draw_detections(np_img_orig, bboxes, classes, self.classes, conf)
        #     self.vis_msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
        #     self.visualization_publisher.publish(self.vis_msg)

        #     cv2.imshow('frame', vis_img)
        #     cv2.waitKey(1)

            
    def timer_callback(self):
        if self.sub_img_flag == True:
            #print("ok")
            # self.YOLOv7_detect()
            a=1

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data)
        self.sub_img_flag = True
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BayerBG2BGR)

        self.YOLOv7_detect()

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# 카메라 두 개를 직접 켜서 detect
# 실제 amz미션에서 사용할 yolo v7 코드

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
        self.declare_parameter("conf_thres", 0.5, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.5, ParameterDescriptor(description="IOU threshold"))
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

        self.video_capture_left = cv2.VideoCapture("/dev/v4l/by-id/usb-046d_罗技高清网络摄像机_C930c_744E4AAE-video-index0", cv2.CAP_V4L2)
        self.video_capture_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.video_capture_right = cv2.VideoCapture("/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_964D8E9E-video-index0", cv2.CAP_V4L2)
        self.video_capture_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # self.video_capture_left = cv2.VideoCapture("/dev/v4l/by-id/usb-046d_罗技高清网络摄像机_C930c_744E4AAE-video-index0", cv2.CAP_V4L2)
        # self.video_capture_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.video_capture_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Timer callback
        self.frequency = 50   # Hz : 1초에 20번 타이머 콜백 실행
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for result image
        # ccm = QoSProfile(reliability = rclpy.qos.ReliabilityPolicy.BEST_EFFORT, depth=5)

        self.corrected_publisher = self.create_publisher(Image, "/corrected_image", 1)
        self.corrected_msg = Image()

        # self.visualization_publisher = self.create_publisher(Image, "/yolo_result", 1)
        # self.vis_msg = Image()

        # Publishers for detection
        self.detect_publisher = self.create_publisher(Detection2DArray, "/yolo_detect", 1)
        # self.detect_msg = Detection2DArray()

        # Realsense package
        self.bridge = CvBridge()
        # qos_profile_sensor_data
        # Subscribers
        # self.sub_img = self.create_subscription(Image, 'video1', self.img_callback, qos_profile_sensor_data)

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
    
    def calculate_iou(self, box1, box2):
        # box1, box2는 [x1, y1, x2, y2] 형식의 바운딩 박스입니다.
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 겹치는 영역 계산
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        # 겹치는 영역의 넓이 계산
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # 각 바운딩 박스의 넓이 계산
        area1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

        # IOU 계산
        iou = intersection_area / (area1 + area2 - intersection_area)

        return iou







    def YOLOv7_detect(self):
        # message refresh

        pub_time_ = self.get_clock().now().to_msg()
        
        ret_right, frame_right = self.video_capture_right.read()
        ret_left, frame_left = self.video_capture_left.read()
        ## stamp
        # pub_time_ = self.get_clock().now().to_msg()


        if ret_right and ret_left:
            # vconcat
            np_img_orig = cv2.vconcat([frame_right, frame_left])
            # cv2.imshow('frame1', np_img_orig)
            # cv2.waitKey(1)
            ## image publish
            self.corrected_msg = self.bridge.cv2_to_imgmsg(np_img_orig, "bgr8")
            self.corrected_msg.header.stamp = pub_time_
            self.corrected_publisher.publish(self.corrected_msg)
            
        # np_img_orig = self.image.copy()

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
        # detections = detections.to("cpu")

        if len(detections) == 0:
            # print("nothing_detect")
            a=1

        else:
            # iou_threshold = 0.1  # 필요에 따라 임계값 조정

            # # 제거할 바운딩 박스를 추적하기 위한 목록 생성
            # remove_indices = []

            # # 바운딩 박스를 반복하여 비교
            # for i in range(len(detections)):
            #     for j in range(i + 1, len(detections)):
            #         iou = self.calculate_iou(detections[i, :4], detections[j, :4])
            #         if iou > iou_threshold:
            #             # 두 바운딩 박스가 겹칠 때 크기가 작은 바운딩 박스를 제거 대상으로 표시
            #             size_i = (detections[i, 2] - detections[i, 0] + 1) * (detections[i, 3] - detections[i, 1] + 1)
            #             size_j = (detections[j, 2] - detections[j, 0] + 1) * (detections[j, 3] - detections[j, 1] + 1)
            #             if size_i < size_j:
            #                 remove_indices.append(i)
            #             else:
            #                 remove_indices.append(j)

            # # 중복된 인덱스를 제거 (중복으로 표시된 바운딩 박스 제거)
            # remove_indices = list(set(remove_indices))

            # # 표시된 제거 대상을 제거
            # detections = np.delete(detections, remove_indices, axis=0)


            #### detection2darray로 메세지 보내기 o ####
            box_info = [[float(x1), float(y1), float(x2), float(y2), float(conf), int(c)]
                        for x1, y1, x2, y2, conf, c in detections[:,:].tolist()]
            
            my_bbox = Detection2DArray()
            my_bbox.header.stamp = pub_time_

            for bbox_coords in box_info:
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

        # image publish
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

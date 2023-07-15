
import sys
sys.path.append('/home/ccm/ros2_ws/src/object_detection/object_detection')
import os
from typing import Tuple, Union, List, Optional
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
import cv_bridge

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from visualizer import draw_detections
from ros import create_detection_msg


def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    """Rescale the output to the original image shape
    :param ori_shape: original width and height [width, height].
    :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
        [num_boxes, >=4], where the first 4 entries of each element have to be
        [x1, y1, x2, y2].
    :param target_shape: target width and height [width, height].
    """
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]

    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale

    return boxes

class ObjectDetection(Node):
    def __init__(self, img_size: Union[Tuple[int, int], None] = (640, 640)):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter("weights", "guide_dog.pt", ParameterDescriptor(description="Weights file")) # weight라는 파라미터를 선언한다. 이 파라미터는 기본값으로 guide_dog.pt를 가지며, 파라미터 설명란에 weights file이라는 설명이 포함된다.
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "cpu", ParameterDescriptor(description="Name of the device"))
        #self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))


        self.weights = self.get_parameter("weights").get_parameter_value().string_value # self.weights 변수에 weights 파라미터의 문자열 값이 할당된다.
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        #self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value

        self.img_size = img_size
        self.image = None

        # Flags
        self.image_bool = False
        #self.camera_RGB = False
        #self.camera_depth = False

        # Timer callback
        self.frequency = 20  # Hz, 1초이 20번의 타이머 콜백 실행
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)   # create_timer함수를 사용해 타이머 생성, 타이머 콜백함수인 self.time_callback 지정.
                                                                                # 1/self.frequency는 타이머의 주기를 나타냄(20Hz)

        # Publishers for Classes
        self.pub_result = self.create_publisher(Image, "/person", 10)
        self.result_msg = Image()


        # Realsense package
        self.bridge = CvBridge()

        
        
        # Subscribers
        self.sub_image = self.create_subscription(Image, '/video1', self.Image_callback, 10)
        self.sub_image # prevent unused variable warning

        set_logging() # YOLOv7 utils.general에서 정의된 함수
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1



    def Image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data)
        self.image_bool = True
    

    # def YOLOv7_detect(self):
    #     """ Preform object detection with YOLOv7"""

    #     # Flip image
    #     #img = cv2.flip(cv2.flip(np.asanyarray(self.image),0),1) # Camera is upside down on the Go1
    #     img = self.image
        
    #     im0 = img.copy()
    #     #img = img[np.newaxis, :, :, :]
    #     img = np.stack(img, 0)
    #     img = img[..., ::-1].transpose((0, 3, 1, 2)) # 나중에 색상영역에 대한 문제가 생기면 이게 문제가 될거같다.
    #     img = np.ascontiguousarray(img)
    #     img = torch.from_numpy(img).to(self.device)
    #     img = img.half() if self.half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)

    #     # Warmup
    #     if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
    #         self.old_img_b = img.shape[0]
    #         self.old_img_h = img.shape[2]
    #         self.old_img_w = img.shape[3]
    #         for i in range(3):
    #             self.model(img)[0]

    #     # Inference
    #     t1 = time_synchronized()
    #     with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
    #         pred = self.model(img)[0]
    #     t2 = time_synchronized()

    #     # Apply NMS
    #     pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
    #     t3 = time_synchronized()

    #     # Process detections   
    #     for i, det in enumerate(pred):  # detections per image
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class

    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 label = f'{self.names[int(cls)]} {conf:.2f}'

    #                 if conf > 0.8: # Limit confidence threshold to 80% for all classes
    #                     # Draw a boundary box around each object
    #                     plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

    #         cv2.imshow("YOLOv7 Object detection result RGB", cv2.resize(im0, None, fx=1.5, fy=1.5))
    #         # if self.use_depth == True:
    #         #     cv2.imshow("YOLOv7 Object detection result Depth", cv2.resize(self.depth_color_map, None, fx=1.5, fy=1.5))
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
############################################################################
    def YOLOv7_detect(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape
        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis=2)
            c = 3

        # automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        # conversion to torch tensor (copied from original yolov7 repo)
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)

        # inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections[:, :4] = rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        # publishing
        if len(detections) == 0:
            print("nothing_detect")
            # detect_length_msg = len(detections)
            # self.detect_length_publisher.publish(detect_length_msg)
            flag_msg = False
            self.flag_publisher.publish(flag_msg)
        else:
            # print(detections)
            print(len(detections))
            detection_msg = create_detection_msg(img_msg, detections)
            detect_length_msg = len(detections)
            self.detect_length_publisher.publish(detect_length_msg)
            flag_msg = True
            self.flag_publisher.publish(flag_msg)
            self.detection_publisher.publish(detection_msg)    

            box_position = [[int(x1), int(y1), int(x2), int(y2)]
                            for x1, y1, x2, y2 in detections[:, :4].tolist()]
            
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes,self.class_labels)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            self.visualization_publisher.publish(vis_msg)

#################################################################################
    def timer_callback(self):
        if self.image_bool == True:
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

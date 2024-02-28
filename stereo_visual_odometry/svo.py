#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import cv2

class StereoVisualOdometry(Node):
    def __init__(self):
        super().__init__("stereo_node")

        self.bridge = CvBridge()
        self.left_img_sub = message_filters.Subscriber(self, Image, '/camera2/left/image_raw')
        self.right_img_sub = message_filters.Subscriber(self, Image, '/camera3/right/image_raw')
        self.left_cam_info = message_filters.Subscriber(self, CameraInfo, "/camera2/left/camera_info")
        self.right_cam_info = message_filters.Subscriber(self, CameraInfo, "/camera3/right/camera_info",)
        ts = message_filters.TimeSynchronizer([self.left_img_sub, self.right_img_sub, self.left_cam_info, self.right_cam_info], 10)
        ts.registerCallback(self.callback)
        self.odom_sub = self.create_subscription(Odometry, "/car/base/odom",  self.odom_callback, 10)
        
        self.focal_length = 477.0
        self.baseline = 70 # 0.07 m
        self.detector = 'sift'

    def callback(self, left_img_msg, right_img_msg, left_cam_info_msg, right_cam_info_msg):
        # self.get_logger().info("stereo node has been started")

        return

    def odom_callback(self, msg):

        return

def main(args=None):
    rclpy.init(args=args)
    node = StereoVisualOdometry()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
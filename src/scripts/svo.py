#!/usr/bin/env python2.7

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
import cv2

class StereoVisualOdometry:
    def __init__(self):
        rospy.init_node('stereo_visual_odometry_node', anonymous=True)
        self.bridge = CvBridge()
        self.left_img_sub = message_filters.Subscriber('/car_1/camera/left/image_raw/compressed', CompressedImage)
        self.right_img_sub = message_filters.Subscriber('/car_1/camera/right/image_raw/compressed', CompressedImage)
        ts = message_filters.TimeSynchronizer([self.left_img_sub, self.right_img_sub], 10)
        ts.registerCallback(self.callback)
        self.odom_sub = rospy.Subscriber("/car_1/base/odom", Odometry, self.odom_callback)
        self.odom_pub = rospy.Publisher('/car_1/odom', Odometry, queue_size=10)
        self.odom_path_publisher = rospy.Publisher('/odom/path', Path, queue_size=10)
        self.path_msg = Path()
        self.p_msg = Path()
        self.path_msg.header.frame_id = "map"
        self.focal_length = 477.0
        self.baseline = 70 # 0.07 m

    def left_img_callback(self, data):
        left_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

    def right_img_callback(self, data):
        right_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    def detect_features(self, img):
        pass

    def track_features(self, prev_img, curr_img, prev_pts):
        pass

    def compute_odometry(self, prev_img, curr_img):
        pass

    def callback(self, left_img_msg, right_img_msg):

        return
    
    def odom_callback(self, odom):
        pose = PoseStamped()
        pose.pose = odom.pose.pose
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        self.p_msg.poses.append(pose)
        self.p_msg.header.frame_id = "map"
        self.odom_path_publisher.publish(self.p_msg)
        return

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rospy.spin()
            rate.sleep()

if __name__ == '__main__':
    svo = StereoVisualOdometry()
    try:
        svo.run()
    except rospy.ROSInterruptException:
        pass

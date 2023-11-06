#!/usr/bin/env python2.7

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
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
        self.detector = 'sift'

    # <----FEATURE EXTRACTOR---->
    def extract_features(self, image, detector, mask=None):
        if detector == 'sift':
            create_detector = cv2.SIFT_create()
        elif detector == 'orb':
            create_detector = cv2.ORB_create()
        keypoints, descriptors = create_detector.detectAndCompute(image, mask)
        return keypoints, descriptors


    # <----FEATURE MATCHER---->
    def match_features(self, first_descriptor, second_descriptor, detector, k=2,  distance_threshold=1.0):
        if detector == 'sift':
            feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = feature_matcher.knnMatch(first_descriptor, second_descriptor, k=k)

        # Filtering out the weak features
        filtered_matches = []
        for match1, match2 in matches:
            if match1.distance <= distance_threshold * match2.distance:
                filtered_matches.append(match1)
        return filtered_matches
    
    def detect_features(self, img):

        pass

    def track_features(self, prev_img, curr_img, prev_pts):
        pass

    def compute_odometry(self, prev_img, curr_img):
        pass

    def callback(self, left_img_msg, right_img_msg):
        left_img = self.bridge.compressed_imgmsg_to_cv2(left_img_msg, "bgr8")
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.bridge.compressed_imgmsg_to_cv2(right_img_msg, "bgr8")
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        # kpLA, dLA = self.extract_features(left_img, self.detector, mask=None)
        print(cv2.__version__)
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

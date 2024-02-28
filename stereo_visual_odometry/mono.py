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
import tf2_ros
import os 

my_arr = []

class MonoVisualOdometry(Node):
    def __init__(self):
        super().__init__("mono_node")
        self.bridge = CvBridge()

        self.left_img_sub = message_filters.Subscriber(self, Image, '/camera2/left/image_raw')
        self.right_img_sub = message_filters.Subscriber(self, Image, '/camera3/right/image_raw')
        self.left_cam_info = message_filters.Subscriber(self, CameraInfo, "/camera2/left/camera_info")
        self.right_cam_info = message_filters.Subscriber(self, CameraInfo, "/camera3/right/camera_info",)
        ts = message_filters.TimeSynchronizer([self.left_img_sub, 
                                               self.right_img_sub, 
                                               self.left_cam_info, 
                                               self.right_cam_info], 10)
        ts.registerCallback(self.callback)

        self.odom_sub = self.create_subscription(Odometry, "/car/base/odom",  self.odom_callback, 10)

        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher()
        
        self.focal_length = 477.0
        self.baseline = 70 # 0.07 m
        self.detector = 'sift'
        self.k_l = np.array([[float(7.188560000000e+02), 0, float(6.071928000000e+02)],
                             [0, float(7.188560000000e+02), float(1.852157000000e+02)],
                             [0, 0, 1]])
        self.k_r = np.array([[float(7.188560000000e+02), 0, float(6.071928000000e+02)],
                             [0, float(7.188560000000e+02), float(1.852157000000e+02)],
                             [0, 0, 1]])

        self.prev_img = None
        self.prev_pts = None
        self.prev_des = None
        self.prev_img_update = False
        self.counter = 0

        self.car_rot = np.eye(3)
        self.car_pos = np.zeros((3))
        self.path_publisher = self.create_publisher(Path, '/car/base/mono_path', 10)
        self.p_msg = Path()

        self.br = tf2_ros.TransformBroadcaster(self)

    def callback(self, left_img_msg, right_img_msg, left_cam_info_msg, right_cam_info_msg):
        prev_img = self.bridge.imgmsg_to_cv2(left_img_msg)
        kps, des = self.sift.detectAndCompute(prev_img, None)

        if not self.prev_img_update:
            self.prev_img = left_img_msg
            self.prev_pts = kps
            self.prev_des = des
            self.prev_img_update = True
            return
        
        pres_img = self.bridge.imgmsg_to_cv2(left_img_msg)
        kps_p, des_p = self.sift.detectAndCompute(pres_img, None)

        matches = self.flann.knnMatch(des_p, self.prev_des, k=2)

        good_matches = []
        for kp_l,kp_r in matches:
            # if kp_l.distance < 0.3*kp_r.distance:
            good_matches.append(kp_l)
        # self.get_logger().info(f"len of matches-{len(matches)}")
        src = np.float32([kps[m.queryIdx].pt for m in good_matches])
        dst = np.float32([self.prev_pts[m.trainIdx].pt for m in good_matches])
        # self.get_logger().info(f"len of src-{len(src)}")
        src = src.reshape(-1, 1, 2)
        dst = dst.reshape(-1, 1, 2)
        # self.get_logger().info(f"shape of src-{src.shape}")
        E, mask = cv2.findEssentialMat(src, dst, self.k_l, cv2.RANSAC, threshold = 5, prob = 0.99, mask = None)
        # self.get_logger().info(f"Essential matrix-{E}")
        # if E is None:
        #     print("Essential Matrix (when shape is incorrect):")
        #     pass
            
        # # print("Shape of E:", E.shape)

        _, R, T, _ = cv2.recoverPose(E, src, dst, self.k_l, mask)

        # self.get_logger().info(f'Rotation Matrix-{R}')
        # self.get_logger().info(f'Translational Matrix-{T}')

        

        transformation_mtx = np.eye(4)

        transformation_mtx[0:3, 0:3] = R
        transformation_mtx[:3, -1] = T[:, 0]

        my_arr.append(transformation_mtx)
        

        # self.get_logger().info(f'Transeformation Matrix-{transformation_mtx}')
        delta = np.dot( self.car_rot, T) 
        self.car_pos[0] += delta[0]
        self.car_pos[1] += delta[1]
        self.car_pos[2] += delta[2]
        
        self.car_rot = np.dot(self.car_rot, R) #R.dot(self.car_rot)
        
        q = self.rotation_mtx_to_quaternion(R)

        self.publish_path(q, T)

        self.prev_img = left_img_msg
        self.prev_pts = kps_p
        self.prev_des = des_p
        return
    
    def publish_path(self, q, T):
        pose= PoseStamped()
        pose.pose.position.x = float(T[0])
        pose.pose.position.y = float(T[1])
        pose.pose.position.z = float(T[2])

        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])

        self.get_logger().info(f'Transeformation Matrix-{pose}')

        pose.header.frame_id = "odom"
        self.p_msg.poses.append(pose)
        self.p_msg.header.frame_id = "map"
        self.path_publisher.publish(self.p_msg)
        return
    
    def odom_callback(self, msg):
        # self.get_logger().info(f"called")
        return
     
    def rotation_mtx_to_quaternion(self, m):
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[0] = 0.5 * t
            t = 0.5/t
            q[1] = (m[2,1] - m[1,2]) * t
            q[2] = (m[0,2] - m[2,0]) * t
            q[3] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[0] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t
        return q


def main(args=None):
    rclpy.init(args=args)
    node = MonoVisualOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:

        rclpy.shutdown()

if __name__ == '__main__':
    main()
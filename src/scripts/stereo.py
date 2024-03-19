#!/usr/bin/env python3

import rospy
import cv2
import tf
import numpy as np
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage  
import rospy
from nav_msgs.msg import Odometry
import tf
from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo


# <!-- <node pkg="rviz" type="rviz" name="rviz" args="-d $(find stereo_visual_odometry)/src/rviz/stereo_config.rviz"/> -->
# <!-- <node pkg="rosbag" type="play" name="player" output="screen" args="--clock /home/bharadwajsirigadi/Datasets/Edge-SLAM-Datasets/KITTI_Datasets/rosbags/kitti_odometry_sequence_00.bag"/> -->

class StereoVO():
    def __init__(self, left_p, right_p):
        self.l_p = np.array(left_p).reshape(3, 4)
        self.r_p = np.array(right_p).reshape(3, 4)
        self.l_k = np.array(left_p).reshape(3, 4)[:3, :3]
        self.r_k = np.array(right_p).reshape(3, 4)[:3, :3]
        self.l_fc = self.l_k[0, 0]
        self.r_fc = self.r_k[0, 0]
        self.l_pp = (self.l_k[0, 2], self.l_k[1, 2])
        self.r_pp = (self.r_k[0, 2], self.r_k[1, 2])
        self.baseline = 54 # 0.07 meters

        self.left_img_pub = rospy.Publisher('/car_1/left_img_rectified', Image, queue_size=10)
        self.right_img_pub = rospy.Publisher('/car_1/right_img_rectified', Image, queue_size=10)

        self.bridge = CvBridge()

        return
    
    def publish_images(self, left_img_rectified, right_img_rectified):
        left_img_msg = self.bridge.cv2_to_imgmsg(left_img_rectified, encoding="bgr8")
        right_img_msg = self.bridge.cv2_to_imgmsg(right_img_rectified, encoding="bgr8")

        left_img_msg.header.stamp = rospy.Time.now()
        right_img_msg.header.stamp = rospy.Time.now()

        self.left_img_pub.publish(left_img_msg)
        self.right_img_pub.publish(right_img_msg)
        return
    
    def filter_imgs(self, left_img, right_img):
        # Input: Left and right images
        # Returns: Rectified left and right images
        # stereo rectification
        blur_left = cv2.bilateralFilter(left_img,9,75,75)
        blur_right = cv2.bilateralFilter(right_img,9,75,75)
        
        self.publish_images(blur_left, blur_right)
        return 
    
    def stereoVO(self, frame_0, frame_1 , MIN_NUM_FEAT):
        # Input: Two image frames of each (left and right)
        # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
        prev_left_img = frame_0[0]
        prev_right_img = frame_0[1]
        pres_left_img = frame_1[0]
        pres_right_img = frame_1[1]

        self.filter_imgs(prev_left_img, prev_right_img)

        return


class Stereo():
    def __init__(self):
        rospy.init_node('stereo', anonymous=True)

        self.k = np.array([[476.7030836014194, 0.0, 400.5],
                           [0.0, 476.7030836014194, 400.5],
                           [0.0, 0.0, 1.0]])
        self.fc = float(476.7030836014194)
        self.pp = (400.5, 400.5)

        # self.fc = 718.8560
        # self.pp = (607.1928, 185.2157)
        # self.k = None
        # self.fc = None  
        # self.pp = None
        self.l_p = None
        self.r_p = None

        self.bridge = CvBridge()

        css = TimeSynchronizer([Subscriber('/kitti/camera_color_left/camera_info', CameraInfo),
                               Subscriber('/kitti/camera_color_right/camera_info', CameraInfo)],queue_size=10)
        css.registerCallback(self.cam_info_callback)

        iss = TimeSynchronizer([Subscriber('/kitti/camera_color_left/image', Image),
                               Subscriber('/kitti/camera_color_right/image', Image)], queue_size=10)
        iss.registerCallback(self.img_callback)

        rospy.Subscriber('/tf', TFMessage, self.tf_callback)

        self.odom_pub = rospy.Publisher('/car_1/odom', Odometry, queue_size=10)
        self.path_publisher = rospy.Publisher('/car/base/mono_path', Path, queue_size=10)

        self.br = tf.TransformBroadcaster()
        self.p_msg = Path()
        
        self.car_rot = np.eye(3)
        self.car_pos = np.zeros((3))

        self.prev_left_img = None
        self.prev_right_img = None
        self.prev_img_update = False
        self.odom_updated = False
        self.cam_info_updated = False
        return
    
    def odom_update(self, data):
        if not self.odom_updated:
            self.car_pos[0] = data.pose.pose.position.x
            self.car_pos[1] = data.pose.pose.position.y
            self.car_pos[2] = data.pose.pose.position.z
            rospy.loginfo('Initial odom updated')
            quat = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
            self.car_rot = tf.transformations.quaternion_matrix(quat)[:3, :3]
            self.odom_updated = True
            return
        return
    
    def quaternion_to_rotation_matrix(self, quaternion):
        # Convert quaternion to rotation matrix
        rotation_matrix = np.eye(3)
        x, y, z, w = quaternion
        rotation_matrix[0, 0] = 1 - 2 * (y**2 + z**2)
        rotation_matrix[0, 1] = 2 * (x*y - z*w)
        rotation_matrix[0, 2] = 2 * (x*z + y*w)
        rotation_matrix[1, 0] = 2 * (x*y + z*w)
        rotation_matrix[1, 1] = 1 - 2 * (x**2 + z**2)
        rotation_matrix[1, 2] = 2 * (y*z - x*w)
        rotation_matrix[2, 0] = 2 * (x*z - y*w)
        rotation_matrix[2, 1] = 2 * (y*z + x*w)
        rotation_matrix[2, 2] = 1 - 2 * (x**2 + y**2)
        return rotation_matrix

    def tf_callback(self, data):
        if not self.odom_updated:
            for tf in data.transforms:
                self.car_pos[0] = (-1) * tf.transform.translation.x
                self.car_pos[1] = (-1) * tf.transform.translation.z
                self.car_pos[2] = (-1) * tf.transform.translation.y
                quat = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z,  tf.transform.rotation.w]
                self.car_rot = self.quaternion_to_rotation_matrix(quat)
            self.odom_updated = True
        return
        
    def publish_odometry(self, car_pos, car_rot):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "car_1_base_link"

        odom.pose.pose.position.x = car_pos[0]
        odom.pose.pose.position.y = car_pos[1]
        odom.pose.pose.position.z = car_pos[2]

        H = np.eye(4)
        H[:3, :3] = car_rot
        #print(self.car_pos)
        H[0, 3] = car_pos[0]
        H[1, 3] = car_pos[1]
        H[2, 3] = car_pos[2]

        quat = tf.transformations.quaternion_from_matrix(H)

        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        # rospy.loginfo('odometry published')
        self.br.sendTransform((car_pos[0], car_pos[1], car_pos[2]),
                              quat,
                              rospy.Time.now(),
                              'car_1_base_link',
                              "map")
        self.odom_pub.publish(odom)
        self.publish_path(quat, car_pos)
        return
    
    def publish_path(self, q, T):
        pose= PoseStamped()
        pose.pose.position.x = -float(T[0])
        pose.pose.position.y = -float(T[2])
        pose.pose.position.z = -float(T[1])
        # pose.pose.position.x = float(T[0])
        # pose.pose.position.y = float(T[1])
        # pose.pose.position.z = float(T[2])

        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])

        pose.header.frame_id = "map"
        self.p_msg.poses.append(pose)
        self.p_msg.header.frame_id = "map"
        self.path_publisher.publish(self.p_msg)
        return
    
    def cam_info_callback(self, left_msg, right_msg):
        if not self.cam_info_updated:
            self.l_p = left_msg.P
            self.r_p = right_msg.P
            # self.l_k = np.array(left_msg.P).reshape(3, 4)[:3, :3]
            # self.r_k = np.array(right_msg.P).reshape(3, 4)[:3, :3]

            self.cam_info_updated = True
            rospy.loginfo('camera info updated')
        return
    
    def img_callback(self, left_img_msg, right_img_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_img_msg)
        right_img = self.bridge.imgmsg_to_cv2(right_img_msg)

        if not self.prev_img_update:
            rospy.loginfo('previous images updated')
            self.prev_left_img = left_img
            self.prev_right_img = right_img
            self.prev_img_update = True
            return
        # rospy.loginfo('got the images')

        prev_frame = [self.prev_left_img, self.prev_right_img]
        pres_frame = [left_img, right_img]
        self.StereoVO = StereoVO(self.l_p, self.r_p)
        self.StereoVO.stereoVO(prev_frame, pres_frame, 1500)


        # previous images update
        self.prev_left_img = left_img
        self.prev_right_img = right_img
        return
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.odom_updated:
                rospy.loginfo('odom not yet updated!')
                rate.sleep()
            if not self.cam_info_updated:
                rospy.loginfo('camera info not yet updated!')
                rate.sleep()
            rospy.loginfo('called')
            rospy.spin()
            rate.sleep()

def main():
    stereo = Stereo()
    try:
        stereo.run()
    except rospy.ROSInterruptException:
        pass

if __name__=="__main__":
    main()
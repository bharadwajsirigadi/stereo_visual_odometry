#!/usr/bin/env python3

import rospy
import cv2
import tf
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage  
import rospy
from nav_msgs.msg import Odometry
import tf

# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [476.7030836014194, 0.0, 400.5, 0.0, 476.7030836014194, 400.5, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [476.7030836014194, 0.0, 400.5, -0.0, 0.0, 476.7030836014194, 400.5, 0.0, 0.0, 0.0, 1.0, 0.0]
# binning_x: 0
# binning_y: 0

class mono_vo():
    def __init__(self, fc, pp:tuple, k:np.array):
        self.fc = fc
        self.pp = pp
        self.k = k
        return
    
    def featureTracking(self, image_1, image_2, points_1):
        # Set Lucas-Kanade Params
        lk_params = dict(winSize  = (21,21),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # Calculate Optical Flow
        points_2, status, err = cv2.calcOpticalFlowPyrLK(image_1, image_2, points_1, None, **lk_params)
        status = status.reshape(status.shape[0])
        points_1 = points_1[status==1]
        points_2 = points_2[status==1]

        # Return Tracked Points
        return points_1,points_2

    def featureDetection(self):
        # Detect FAST Features
        thresh = dict(threshold=25, nonmaxSuppression=True)
        fast = cv2.FastFeatureDetector_create(**thresh)
        return fast

    def monoVO(self, frame_0, frame_1 , MIN_NUM_FEAT):
        # Input: Two image frames
        # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
        image_1 = frame_0
        image_2 = frame_1

        detector = self.featureDetection()
        kp1      = detector.detect(image_1)
        points_1       = np.array([ele.pt for ele in kp1],dtype='float32')
        points_1, points_2   = self.featureTracking(image_1, image_2, points_1)

        E, mask = cv2.findEssentialMat(points_2, points_1, self.fc, self.pp, cv2.RANSAC,0.999,1.0)
        _, R, t, mask = cv2.recoverPose(E, points_2, points_1,focal=self.fc, pp = self.pp)
        if len(points_2) < MIN_NUM_FEAT:
            return R,t, False
        
        return R,t, True

class Mono_vo():
    def __init__(self, fc, pp:tuple, k:np.array):
        self.fc = fc
        self.pp = pp
        self.k = k

        self.sift = cv2.xfeatures2d.SIFT_create() 
        self.flann = cv2.FlannBasedMatcher()
        return
    
    def monoVO(self, pres_img, prev_img):
        pres_kps, pres_des = self.sift.detectAndCompute(pres_img, None)
        prev_kps, prev_des = self.sift.detectAndCompute(prev_img, None)


        matches = self.flann.knnMatch(pres_des, prev_des, k=2)
        good_matches = []
        for kp_l,kp_r in matches:
            if kp_l.distance < 0.3*kp_r.distance:
                good_matches.append(kp_l)

        src = np.float32([pres_kps[m.queryIdx].pt for m in good_matches])
        dst = np.float32([prev_kps[m.trainIdx].pt for m in good_matches])
        
        src = src.reshape(-1, 1, 2)
        dst = dst.reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(src, dst, self.k, cv2.RANSAC, threshold = 5, prob = 0.99, mask = None)
      
        if E is None:
            rospy.loginfo("Essential Matrix (when shape is incorrect):")
            pass

        _, R, T, _ = cv2.recoverPose(E, src, dst, self.k, mask)
        return R, T

class Monocular():
    def __init__(self):
        rospy.init_node('mono', anonymous=True)

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
                          
        self.bridge = CvBridge()
        rospy.Subscriber('/kitti/camera_color_left/image', Image, self.left_img_callback)
        rospy.Subscriber('/kitti/camera_color_left/camera_info', CameraInfo, self.camera_info_callback)
        # self.intial_odom = rospy.Subscriber('/car_1/base/odom', Odometry, self.odom_update)
        rospy.Subscriber('/tf', TFMessage, self.tf_callback)

        self.odom_pub = rospy.Publisher('/car_1/odom', Odometry, queue_size=10)
        self.path_publisher = rospy.Publisher('/car/base/mono_path', Path, queue_size=10)

        self.br = tf.TransformBroadcaster()
        self.p_msg = Path()
        self.monoVO = mono_vo(self.fc, self.pp, self.k)
        self.MonoVO = Mono_vo(self.fc, self.pp, self.k)

        self.car_rot = np.eye(3)
        self.car_pos = np.zeros((3))

        self.prev_img = None
        self.prev_img_update = False
        self.odom_updated = False
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
    
    def camera_info_callback(self, msg):
        # rospy.loginfo(msg)
        if self.k is None:
            return
        self.k = np.array(msg.K).reshape(3, 3)
        self.fc = self.k[0, 0]
        self.pp = (self.k[0, 2], self.k[1, 2])
        return
    
    def left_img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)

        if not self.prev_img_update:
            rospy.loginfo('previous image updated')
            self.prev_img = img
            self.prev_img_update = True
            return

        R, T, _ = self.monoVO.monoVO(self.prev_img, img, 1500)
        # R, T = self.MonoVO.monoVO(self.prev_img, img)
        rospy.loginfo('got the computation')
        # rospy.loginfo(R)
        transformation_mtx = np.eye(4)
        transformation_mtx[0:3, 0:3] = R
        transformation_mtx[:3, -1] = T[:, 0]

        delta = np.dot(self.car_rot, T) 
        self.car_pos[0] += delta[0]
        self.car_pos[1] += delta[1]
        self.car_pos[2] += delta[2]
        
        self.car_rot = np.dot(self.car_rot, R)
        self.publish_odometry(self.car_pos, self.car_rot)

        # previous image update
        self.prev_img = img
        return
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.odom_updated:
                rospy.loginfo('odom not yet updated!')
                rate.sleep()
            # if self.k is None:
            #     rospy.loginfo('camera info not yet updated!')
            #     rate.sleep()
            rospy.loginfo('called')
            rospy.spin()
            rate.sleep()

def main():
    mono = Monocular()
    try:
        mono.run()
    except rospy.ROSInterruptException:
        pass

if __name__=="__main__":
    main()


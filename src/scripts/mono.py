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


class Monocular():
    def __init__(self):
        rospy.init_node('mono', anonymous=True)

        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher()

        self.k = np.array([[476.7030836014194, 0.0, 400.5],
                           [0.0, 476.7030836014194, 400.5],
                           [0.0, 0.0, 1.0]])
        self.f = float(476.7030836014194)
                          
        self.bridge = CvBridge()
        rospy.Subscriber('/car_1/camera/left/image_raw/compressed', CompressedImage, self.left_img_callback)
        rospy.Subscriber('/car_1/camera/left/camera_info', CameraInfo, self.camera_info_callback)
        self.intial_odom = rospy.Subscriber('/car_1/base/odom', Odometry, self.odom_update)

        self.odom_pub = rospy.Publisher('/car_1/odom', Odometry, queue_size=10)
        self.path_publisher = rospy.Publisher('/car/base/mono_path', Path, queue_size=10)

        self.br = tf.TransformBroadcaster()
        self.p_msg = Path()

        self.car_rot = np.eye(3)
        self.car_pos = np.zeros((3))

        self.prev_img = None
        self.prev_kps = None
        self.prev_des = None
        self.prev_img_update = False
        self.odom_updated = False
        return
    
    def odom_update(self, data):
        if not self.odom_updated:
            self.car_pos[0] = data.pose.pose.position.x
            self.car_pos[1] = data.pose.pose.position.y
            self.car_pos[2] = data.pose.pose.position.z
            rospy.loginfo('odom updated')
            quat = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z , data.pose.pose.orientation.w ]
            self.car_rot = tf.transformations.quaternion_matrix(quat)[:3, :3]
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
        pose.pose.position.x = float(T[0])
        pose.pose.position.y = float(T[1])
        pose.pose.position.z = float(T[2])

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
        return
    
    def left_img_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        pres_kps, pres_des = self.sift.detectAndCompute(img, None)

        if not self.prev_img_update:
            rospy.loginfo('previous image updated')
            self.prev_img = img
            self.prev_kps = pres_kps
            self.prev_des = pres_des
            self.prev_img_update = True
            return
        
        matches = self.flann.knnMatch(pres_des, self.prev_des, k=2)
        good_matches = []
        for kp_l,kp_r in matches:
            if kp_l.distance < 0.3*kp_r.distance:
                good_matches.append(kp_l)

        src = np.float32([pres_kps[m.queryIdx].pt for m in good_matches])
        dst = np.float32([self.prev_kps[m.trainIdx].pt for m in good_matches])
        
        src = src.reshape(-1, 1, 2)
        dst = dst.reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(src, dst, self.k, cv2.RANSAC, threshold = 5, prob = 0.99, mask = None)
      
        # if E is None:
        #     rospy.loginfo("Essential Matrix (when shape is incorrect):")
        #     pass

        _, R, T, _ = cv2.recoverPose(E, src, dst, self.k, mask)
        
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
        self.prev_kps = pres_kps
        self.prev_des = pres_des
        return
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self.odom_updated:
                rospy.loginfo('odom not yet updated!')
                rate.sleep()
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


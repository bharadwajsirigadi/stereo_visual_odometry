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
        self.baseline = 0.54 # 0.07 meters

        self.left_img_pub = rospy.Publisher('/car_1/left_img_rectified', Image, queue_size=10)
        self.right_img_pub = rospy.Publisher('/car_1/right_img_rectified', Image, queue_size=10)

        self.bridge = CvBridge()

        self.R = None
        self.T = None

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
        return blur_left, blur_right
    
    def get_disparity(self, left_img, right_img):
        # Input: Left and right images
        # Returns: Disparity image
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(left_gray, right_gray)
        disparity = np.divide(disparity, 16.0)
        return disparity
    
    def detect_features(self, img): 
        # Input: Image
        # Returns: Keypoints and descriptors
        # sift = cv2.SIFT_create()
        # kp, des = sift.detectAndCompute(img, None)
        featureEngine = cv2.FastFeatureDetector_create()
        TILE_H = 10
        TILE_W = 20
        H,W, _ = img.shape
        kp = []
        idx = 0
        for y in range(0, H, TILE_H):
            for x in range(0, W, TILE_W):
                imPatch = img[y:y+TILE_H, x:x+TILE_W]
                keypoints = featureEngine.detect(imPatch)
                for pt in keypoints:
                    pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

                if (len(keypoints) > 20):
                    keypoints = sorted(keypoints, key=lambda x: -x.response)
                    for kpt in keypoints[0:20]:
                        kp.append(kpt)
                else:
                    for kpt in keypoints:
                        kp.append(kpt)
        # return kp, des
        # keypoints_array = np.array([kp.pt for kp in kp], dtype=np.float32)
        # print(keypoints_array.shape)
        return kp
    
    def track_points(self, prev_img, pres_img, prev_kp):
        # Input: Previous and present images, previous keypoints and descriptors

        trackPoints1 = cv2.KeyPoint_convert(prev_kp)
        trackPoints1 = np.expand_dims(trackPoints1, axis=1)
        # trackPoints1 = prev_kp

        lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(prev_img, pres_img, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)
        # ptTrackable = np.where(st == 1, 1,0).astype(bool)
        ptTrackable = np.where(st == 1, True, False)
        trackPoints1_KLT = trackPoints1[st == 1]
        trackPoints2_KLT_t = trackPoints2[st == 1]
        trackPoints2_KLT = np.around(trackPoints2_KLT_t)

        # among tracked points take points within error measure
        error = 3
        errTrackablePoints = err[ptTrackable, ...]
        errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
        trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
        trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]

        return trackPoints1_KLT, trackPoints2_KLT
    
    def get_3d_points(self, points_1, disparity):
        # Input: Tracked points and disparity image
        # Returns: 3D points
        # Generate 3D point cloud
        points_2d = []
        points_3d = []
        for point in points_1:
            x = int(point[0])
            y = int(point[1])
            if x >= 0 and x < disparity.shape[1] and y >= 0 and y < disparity.shape[0]:
                disparity_val = disparity[y, x]

                if disparity_val > 0:  # Ensure valid disparity value
                    depth = self.l_fc / disparity_val
                    point_3d = [(point[0] - self.l_fc) * depth / self.l_fc,
                                (point[1] - self.l_fc) * depth / self.l_fc,
                                depth]
                    points_3d.append(point_3d)
                    points_2d.append(point)
                else:
                    continue
            else:
                continue

        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)
        H,W = disparity.shape

        # check for validity of tracked point Coordinates
        hPts = np.where(points_2d[:,1] >= H)
        wPts = np.where(points_2d[:,0] >= W)
        outTrackPts = hPts[0].tolist() + wPts[0].tolist()
        outDeletePts = list(set(outTrackPts))

        if len(outDeletePts) > 0:
            points_2d = np.delete(points_2d, outDeletePts, axis=0)
            points_3d = np.delete(points_3d, outDeletePts, axis=0)
        else:
            points_2d = points_2d
            points_3d = points_3d
        return points_3d, points_2d
    
    def get_displaced_points(self, trackPoints1_KLT_L, trackPoints2_KLT_L, ImT1_disparityA, ImT2_disparityA):
        trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
        trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
        selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

        disparityMinThres = 0.0
        disparityMaxThres = 100.0

        for i in range(trackPoints1_KLT_L.shape[0]):
            x1, y1 = int(trackPoints1_KLT_L[i, 0]), int(trackPoints1_KLT_L[i, 1])
            x2, y2 = int(trackPoints2_KLT_L[i, 0]), int(trackPoints2_KLT_L[i, 1])

            if (x1 >= 0 and x1 < ImT1_disparityA.shape[1] and y1 >= 0 and y1 < ImT1_disparityA.shape[0] and
                x2 >= 0 and x2 < ImT2_disparityA.shape[1] and y2 >= 0 and y2 < ImT2_disparityA.shape[0]):
                T1Disparity = ImT1_disparityA[y1, x1]
                T2Disparity = ImT2_disparityA[y2, x2]

                if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres
                and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):



                    if (T1Disparity > 0 and T2Disparity > 0):
                        trackPoints1_KLT_R[i, 0] = x1 - T1Disparity
                        trackPoints2_KLT_R[i, 0] = x2 - T2Disparity
                        selectedPointMap[i] = 1

        selectedPointMap = selectedPointMap.astype(bool)
        trackPoints1_KLT_L_2d = trackPoints1_KLT_L[selectedPointMap]
        trackPoints1_KLT_R_2d = trackPoints1_KLT_R[selectedPointMap]
        trackPoints2_KLT_L_2d = trackPoints2_KLT_L[selectedPointMap]
        trackPoints2_KLT_R_2d = trackPoints2_KLT_R[selectedPointMap]
        
        return trackPoints1_KLT_L_2d, trackPoints1_KLT_R_2d, trackPoints2_KLT_L_2d, trackPoints2_KLT_R_2d
    
    def triangulate(self, points2D_L, points2D_R, Proj1, Proj2):
        numPoints = points2D_L.shape[0]
        d3dPoints = np.ones((numPoints,3))

        for i in range(numPoints):
            pLeft = points2D_L[i,:]
            pRight = points2D_R[i,:]

            X = np.zeros((4,4))
            X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
            X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
            X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
            X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

            [u,s,v] = np.linalg.svd(X)
            v = v.transpose()
            vSmall = v[:,-1]
            vSmall /= vSmall[-1]

            d3dPoints[i, :] = vSmall[0:-1]

        return d3dPoints
    
    def stereoVO(self, frame_0, frame_1 , MIN_NUM_FEAT):
        # Input: Two image frames of each (left and right)
        # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
        prev_left_img = frame_0[0]
        prev_right_img = frame_0[1]
        pres_left_img = frame_1[0]
        pres_right_img = frame_1[1]
        # prev_left = prev_left_img
        # prev_right = prev_right_img
        # pres_left = pres_left_img
        # pres_right = pres_right_img 
        prev_left, prev_right = self.filter_imgs(prev_left_img, prev_right_img)
        pres_left, pres_right = self.filter_imgs(pres_left_img, pres_right_img)
        disparity_prev = self.get_disparity(prev_left, prev_right)
        disparity_pres = self.get_disparity(pres_left, pres_right)

        # kp_prev_l, des_prev_l = self.detect_features(prev_left)
        kp_prev_l = self.detect_features(prev_left)
        points_1, points_2 = self.track_points(prev_left, pres_left, kp_prev_l)


        points_2D_prev_l, points_2D_prev_r, points_2D_pres_l, points_2D_pres_r = self.get_displaced_points(points_1, points_2, disparity_prev, disparity_pres)
        points_3D_prev = self.triangulate(points_2D_prev_l, points_2D_prev_r, self.l_p, self.r_p)
        points_3D_pres = self.triangulate(points_2D_pres_l, points_2D_pres_r, self.l_p, self.r_p)
        print('points_3D Shapes', points_3D_prev.shape, points_3D_pres.shape)
        print('points_2D Shapes', points_2D_prev_l.shape, points_2D_prev_r.shape)

        if points_3D_prev.shape[0] < 7:
            return None, None
        _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3D_prev, points_2D_pres_l, self.l_k, None)
        # # Convert the rotation vector to a rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        # # Convert the translation vector to a numpy array
        T = np.array(tvec).reshape(3)

        return R, T
     
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
        
        R, T = self.StereoVO.stereoVO(prev_frame, pres_frame, 1500)
        if R is None and T is None:
            self.prev_left_img = left_img
            self.prev_right_img = right_img
            return
        # self.StereoVO.stereoVO(prev_frame, pres_frame, 1500)
        # --------------------------------
        transformation_mtx = np.eye(4)
        transformation_mtx[0:3, 0:3] = R
        T = T.reshape(3, 1)
        transformation_mtx[:3, -1] = T[:, 0]

        delta = np.dot(self.car_rot, T) 
        self.car_pos[0] += delta[0]
        self.car_pos[1] += delta[1]
        self.car_pos[2] += delta[2]
        
        self.car_rot = np.dot(self.car_rot, R)
        self.publish_odometry(self.car_pos, self.car_rot)

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
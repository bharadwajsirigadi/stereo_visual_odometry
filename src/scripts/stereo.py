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
from scipy.optimize import least_squares


# <!-- <node pkg="rviz" type="rviz" name="rviz" args="-d $(find stereo_visual_odometry)/src/rviz/stereo_config.rviz"/> -->
# <!-- <node pkg="rosbag" type="play" name="player" output="screen" args="--clock /home/bharadwajsirigadi/Datasets/Edge-SLAM-Datasets/KITTI_Datasets/rosbags/kitti_odometry_sequence_00.bag"/> -->

class StereoVO():
    def __init__(self, left_p, right_p):
        self.l_p = np.array(left_p).reshape(3, 4)
        print(self.l_p)
        self.r_p = np.array(right_p).reshape(3, 4)
        print(self.r_p)
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
        left_gray = left_img
        right_gray = right_img
        stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=11)
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
        disparity = np.divide(disparity, 16.0)
        return disparity
    
    def detect_features(self, img): 
        # Input: Image
        # Returns: Keypoints and descriptors
        # sift = cv2.SIFT_create()
        # kp, des = sift.detectAndCompute(img, None)
        tile_h = 10
        tile_w = 20
        featureEngine = cv2.FastFeatureDetector_create()
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = featureEngine.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10] 
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten
    
    def track_points(self, prev_img, pres_img, prev_kp):
        # Input: Previous and present images, previous keypoints and descriptors

        trackPoints1 = cv2.KeyPoint_convert(prev_kp)
        trackPoints1 = np.expand_dims(trackPoints1, axis=1)
        # trackPoints1 = prev_kp

        lk_params = dict( winSize  = (10,10),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(prev_img, pres_img, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)
        print('trackpoints just after optical flow', len(trackPoints2))
        max_error = 9
        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackPoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackPoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = pres_img.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2
    
    def get_displaced_points(self, trackPoints1_KLT_L, trackPoints2_KLT_L, ImT1_disparityA, ImT2_disparityA):
        min_disp = 0.0
        max_disp = 100.0

        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(trackPoints1_KLT_L, ImT1_disparityA)
        disp2, mask2 = get_idxs(trackPoints2_KLT_L, ImT2_disparityA)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = trackPoints1_KLT_L[in_bounds], trackPoints2_KLT_L[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r
        
        # return trackPoints1_KLT_L_2d, trackPoints1_KLT_R_2d, trackPoints2_KLT_L_2d, trackPoints2_KLT_R_2d
    
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        print("Shapes:", self.l_p.shape, self.r_p.shape, q1_l.shape, q1_r.shape)
        print("Data types:", self.l_p.dtype, self.r_p.dtype, q1_l.T.dtype, q1_r.T.dtype)
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.l_p, self.r_p, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.l_p, self.r_p, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])

        for i in range(len(Q2) - 1, 0, -1):
            if (Q2[i][2] > 75 or Q1[i][2] > 75):
                Q2 = np.delete(Q2, i, axis=0)
                q2_l = np.delete(q2_l,i, axis=0)
                q2_r = np.delete(q2_r,i, axis=0)
                Q1 = np.delete(Q1, i, axis=0)
                q1_l = np.delete(q1_l,i, axis=0)
                q1_r = np.delete(q1_r,i, axis=0)

        return Q1, Q2, q1_l, q1_r, q2_l, q2_r
    
    
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.l_p, transf)
        b_projection = np.matmul(self.l_p, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals
    
    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]

        return R, t
        # # Make the transformation matrix
        # transformation_matrix = self._form_transf(R, t)
        # return transformation_matrix
    
    
    def stereoVO(self, frame_0, frame_1):
        # Input: Two image frames of each (left and right)
        # Returns: Rotation matrix and translation vector, boolean validity, ignore pose if false
        prev_left = frame_0[0]
        prev_right = frame_0[1]
        pres_left = frame_1[0]
        pres_right = frame_1[1]
        # prev_left = prev_left_img
        # prev_right = prev_right_img
        # pres_left = pres_left_img
        # pres_right = pres_right_img 
        prev_left, prev_right = self.filter_imgs(prev_left, prev_right)
        pres_left, pres_right = self.filter_imgs(pres_left, pres_right)
        disparity_prev = self.get_disparity(prev_left, prev_right)
        disparity_pres = self.get_disparity(pres_left, pres_right)

        # kp_prev_l, des_prev_l = self.detect_features(prev_left)
        kp_prev_l = self.detect_features(prev_left)
        print('points_1 just aafter detecting features', len(kp_prev_l))
        points_1, points_2 = self.track_points(prev_left, pres_left, kp_prev_l)
        print('points_1 after tracking features', len(points_1))
        print('points_2 after tracking features', len(points_2))
        
        points_2D_prev_l, points_2D_prev_r, points_2D_pres_l, points_2D_pres_r = self.get_displaced_points(points_1, points_2, disparity_prev, disparity_pres)
        Q1, Q2, q1_l, q1_r, q2_l, q2_r = self.calc_3d(points_2D_prev_l, points_2D_prev_r, points_2D_pres_l, points_2D_pres_r)

        R, T = self.estimate_pose(q1_l, q1_r, Q1, Q2, max_iter=100)
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

        css = TimeSynchronizer([Subscriber('/kitti/camera_gray_left/camera_info', CameraInfo),
                               Subscriber('/kitti/camera_gray_right/camera_info', CameraInfo)],queue_size=10)
        css.registerCallback(self.cam_info_callback)

        iss = TimeSynchronizer([Subscriber('/kitti/camera_gray_left/image', Image),
                               Subscriber('/kitti/camera_gray_right/image', Image)], queue_size=20)
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

        self.StereoVO = None

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
            self.StereoVO = StereoVO(self.l_p, self.r_p)
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

        prev_frame = [self.prev_left_img, self.prev_right_img]
        pres_frame = [left_img, right_img]

        R, T = self.StereoVO.stereoVO(prev_frame, pres_frame)
        if R is None and T is None:
            self.prev_left_img = left_img
            self.prev_right_img = right_img
            return
      
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
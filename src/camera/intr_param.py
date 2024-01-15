import cv2
import numpy as np
import sys, os, time, math
import pyrealsense2 as rs

# Plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from scipy.spatial.transform import Rotation

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
# from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user
from examples.pybullet.utils.pybullet_tools.utils import multiply

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable the color stream
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable the color stream

        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        self.camera_matrix = np.array([
                [self.intrinsics.fx, 0, self.intrinsics.ppx],
                [0, self.intrinsics.fy, self.intrinsics.ppy],
                [0, 0, 1]
            ])
        
        self.dist_coeffs = np.array(self.intrinsics.coeffs)
        # self.intrinsics = np.array(self.intrinsics)


        


    def aruco_detection(self) -> dict: # return a dictionary of aruco id and its center point

        # Define the dictionary and parameters for ArUco marker detection

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # parameters = cv2.aruco.DetectorParameters_create()
        # Initialize the RealSense pipeline
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable the color stream

        # Start the pipeline
        # self.pipeline.start(config)

        try:

            start_time = time.time()

            while True:
                # Wait for the next frame
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert the color frame to a numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Detect ArUco markers
                corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict)#, parameters=parameters)

                if ids is not None:
                    
                    # Draw the detected markers and estimate pose
                    color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, self.camera_matrix, self.dist_coeffs)

                    # print(f'rvecs = {rvecs}')

                    center_dict = {}
                    orientation_dict = {}

                    for i in range(len(ids)):
                        # 
                        central_point = (sum(corner[0] for corner in corners[i][0]) / 4, sum(corner[1] for corner in corners[i][0]) / 4)
                        cv2.circle(color_image, (int(central_point[0]), int(central_point[1])), 50, (255, 255, 255), thickness=4)

                        # Draw axis and coordinate system for each marker
                        cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.1)

                        center_dict[ids[i][0]] = central_point

                        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                        # Extract Euler angles (roll, pitch, yaw) from the rotation matrix
                        # euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                        rotation = Rotation.from_matrix(rotation_matrix)
                        quaternion = rotation.as_quat()
                        orientation_dict[ids[i][0]] = quaternion


                # Display the image with marker detection
                cv2.imshow("ArUco Pose Estimation", color_image)
                
                # Break when it's stable:
                if time.time() - start_time > 20 or cv2.waitKey(1) & 0xFF == 27:
                    break
              

            return center_dict, orientation_dict 

        finally:
            # Stop the RealSense pipeline and release resources
            # self.pipeline.stop()
            cv2.destroyAllWindows()


    def pixel2pose(self, center_dict, orientation_dict, ee_pose=[0,0,0]):
        try:
            pose_dict = {}

            start_time = time.time()
            while True:
                # Wait for the next frame
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Convert the color frame to a numpy array
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                for key in center_dict.keys():
                    depth = depth_frame.get_distance(np.int0(center_dict[key][0]), np.int0(center_dict[key][1]))
                    # Calculate real world coordinates in camera frame
                    d_X = depth*(center_dict[key][0] - self.intrinsics.ppx)/self.intrinsics.fx 
                    d_Y = depth*(center_dict[key][1] - self.intrinsics.ppy)/self.intrinsics.fy
                    d_Z = depth

                    pose_dict[str(key)] = ((d_X, d_Y, d_Z), orientation_dict[key])

                # Break when it's stable:
                if time.time() - start_time > 1 or cv2.waitKey(1) & 0xFF == 27:
                    break

                # Display the image with marker detection
                # cv2.imshow("ArUco Pose Estimation", color_image)

 
            return pose_dict 

        finally:
            # Stop the RealSense pipeline and release resources
            self.pipeline.stop()
            cv2.destroyAllWindows()


 


if __name__ == '__main__':

    # cmtx, dist = read_camera_parameters()
    camera = Camera()

    center_dict, orientation_dict = camera.aruco_detection()

    pose_dict = camera.pixel2pose(center_dict, orientation_dict)

    p_c = pose_dict['11'] # pose of the POINT in the camera frame


    quat_e2w = quaternion_from_euler(math.pi, 0, 0)
    T_e2w = ((0.5, 0.0, 0.55), quat_e2w) # end effector to world

    quat_c2e = quaternion_from_euler(0, 0, math.pi/2)
    T_c2e = ((0.05, 0, 0.05), quat_c2e) # camera to end effector

    p_w = multiply(T_e2w, T_c2e, p_c)



    print(p_w)
    print(np.array(euler_from_quaternion(p_w[1])) * 180 / math.pi)

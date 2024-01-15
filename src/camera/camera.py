import cv2
import numpy as np
import sys, os, time, math, copy
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

import open3d as o3d

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable the color stream
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable the color stream

        
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
        self.pipeline.stop()


    def aruco_pose_detection(self, ee_pose=None) -> dict: # return a dictionary of aruco id and its center point

        # Define the dictionary and parameters for ArUco marker detection

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)

        try:

            start_time = time.time()
            
            pose_dict = {}
            while True:
                # Wait for the next frame
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Convert the color frame to a numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Detect ArUco markers
                detector_parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, detector_parameters)
                # corners, ids, rejected = detector.detectMarkers(color_image)
                corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict)#, parameters=parameters)

                
                if ids is not None:
                    # Draw the detected markers and estimate pose
                    color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, self.camera_matrix, self.dist_coeffs)
         

                    
                    for i in range(len(ids)):
                        # Get point in picture -- pixel
                        central_pix = (sum(corner[0] for corner in corners[i][0]) / 4, sum(corner[1] for corner in corners[i][0]) / 4)
                        cv2.circle(color_image, (int(central_pix[0]), int(central_pix[1])), 50, (255, 255, 255), thickness=4)

                        # ------------------------------------- 
                        # Calculate coordinates in camera frame 
                        # -------------------------------------
                        # Position
                        depth = depth_frame.get_distance(np.int0(central_pix[0]), np.int0(central_pix[1]))
                        # Calculate real world coordinates in camera frame
                        d_X = depth*(central_pix[0] - self.intrinsics.ppx)/self.intrinsics.fx 
                        d_Y = depth*(central_pix[1] - self.intrinsics.ppy)/self.intrinsics.fy
                        d_Z = depth
                        # Orientation
                        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                        rotation = Rotation.from_matrix(rotation_matrix)
                        quaternion = rotation.as_quat()
                        pose_camera = ((d_X, d_Y, d_Z), quaternion)
                        # --------------
                        # To world frame
                        # --------------
                        # End effector to world
                        if ee_pose == None:
                            quat_e2w = quaternion_from_euler(math.pi, 0, 0)
                            T_e2w = ((0.5, 0.0, 0.6), quat_e2w)
                        else:
                            T_e2w = ee_pose
                        # Camera to end effector
                        quat_c2e = quaternion_from_euler(0, 0, math.pi/2)
                        T_c2e = ((0.05, -0.04, 0.05), quat_c2e) # camera to end effector   

                        pose_world = multiply(T_e2w, T_c2e, pose_camera)
                        
                        # Here we only keep the yaw angle
                        pose_word_euler = euler_from_quaternion(pose_world[1])
                        pose_world_quat = quaternion_from_euler(0, 0, pose_word_euler[2])
                        pose_world = (pose_world[0], pose_world_quat)

                        pose_dict[str(ids[i][0])] = pose_world


                        # Draw axis and coordinate system for each marker
                        cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.01)


                

                # Display the image with marker detection
                cv2.imshow("ArUco Pose Estimation", color_image)
                
                # Break when it's stable:
                if time.time() - start_time > 2 or cv2.waitKey(1) & 0xFF == 27:
                    self.pipeline.stop()
                    break
              

            return pose_dict

        finally:
            # Stop the RealSense pipeline and release resources
            # self.pipeline.stop()
            cv2.destroyAllWindows()

    ######################################################################

    def depth_mask(self) -> np.array: # return a dictionary of aruco id and its center point
        self.pipeline.start(self.config)

        align = rs.align(rs.stream.color)

        start_time = time.time()

        while True:
            # Wait for the next frame
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            # color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame:
                continue

            # Convert the color frame to a numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)
            depth_image = cv2.resize(depth_image, (64, 48))

            base = depth_image[int(depth_image.shape[0]/2), int(depth_image.shape[1]/2)]

            depth_masked = np.zeros((depth_image.shape[0], depth_image.shape[1]))
            for w in range(0, depth_image.shape[0]):
                for h in range(0, depth_image.shape[1]):
                    depth = depth_image[w, h]
                    depth_masked[w, h] = 2 if (depth < base + 20) and (depth > 0) else 0 
                    depth_masked[w, h] = 1 if (depth < base + 50) and (depth > base + 20) else depth_masked[w, h]

            # plt.imshow(depth_masked)
            # plt.show()        
            # Display the image with marker detection
            # cv2.imshow("depth", depth_masked)
            
            # Break when it's stable:
            if time.time() - start_time > 3 or cv2.waitKey(1) & 0xFF == 27:
                self.pipeline.stop()
                break

        return depth_masked #depth_masked
    
    #################################################################################

    def reconstruct(self, file_name='hull'): # depth_from_base (mm)
       
        self.pipeline.start(self.config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            for fid in range(20):
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

            t0 = time.time()

            frames = self.pipeline.wait_for_frames()
            t1 = time.time()
            aligned_frames = align.process(frames)

            profile = aligned_frames.get_profile()
            intrinsics = profile.as_video_stream_profile().intrinsics

            # Rs intrinsics to open3d intrinsics
            pinhole_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width, intrinsics.height, intrinsics.fx,
                intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
            
            depth_image_rs = np.asanyarray(aligned_frames.get_depth_frame().get_data())
            color_image_rs = np.asanyarray(aligned_frames.get_color_frame().get_data())

            depth_image_rs_object = copy.deepcopy(depth_image_rs)
            depth_image_rs_background = copy.deepcopy(depth_image_rs)

            depth_base = depth_image_rs[int(depth_image_rs.shape[0]/2), int(depth_image_rs.shape[1]/2)]

            for w in range(0, depth_image_rs.shape[0]):
                for h in range(0, depth_image_rs.shape[1]):
                    depth = depth_image_rs[w, h]
                    depth_image_rs_object[w, h] = depth_image_rs[w, h] if (depth < depth_base + 15) and (depth > 0) else 0 
                    depth_image_rs_background[w, h] = depth_image_rs[w, h] if (depth > depth_base + 15) and (depth < depth_base + 50) else 0

          
            color_image = o3d.geometry.Image(color_image_rs)
            depth_image_object = o3d.geometry.Image(depth_image_rs_object)
            depth_image_background = o3d.geometry.Image(depth_image_rs_background)

            
            # Create rgbd image from color and depth images
            rgbd_image_object = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image_object)
            rgbd_image_background = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image_background)
    
            
            # -------------------------------------
            # Create point cloud from rgbd image
            pcd_object_top = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image_object, pinhole_camera_intrinsics)
            pcd_object_bottom = copy.deepcopy(pcd_object_top).translate(np.array([0, 0, 0.02]))
            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_object_top.points), np.asarray(pcd_object_bottom.points)), axis=0))

            pcd_background_top = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image_background, pinhole_camera_intrinsics)
            pcd_background_bottom = copy.deepcopy(pcd_background_top).translate(np.array([0, 0, 0.05]))
            pcd_background = o3d.geometry.PointCloud()
            pcd_background.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd_background_top.points), np.asarray(pcd_background_bottom.points)), axis=0))
            
            # -------------------------------------
            # Flip it, otherwise the pointcloud will be upside down
            pcd_object.transform([[1, 0, 0, 0], [0, -1, 0, 0], 
                            [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_background.transform([[1, 0, 0, 0], [0, -1, 0, 0], 
                            [0, 0, -1, 0], [0, 0, 0, 1]])
            
            # pcd_object = pcd_object.voxel_down_sample(voxel_size=0.01)
            # pcd_background = pcd_background.voxel_down_sample(voxel_size=0.01)


            pcd_object.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd_background.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            center = pcd_object.get_center()

            # Convex hull
            hull_object, _ = pcd_object.compute_convex_hull()
            hull_background, _ = pcd_background.compute_convex_hull()

            # Transform
            hull_object_trans = copy.deepcopy(hull_object).translate(-np.array(center))
            hull_background_trans = copy.deepcopy(hull_background).translate(-np.array(center))

            # o3d.visualization.draw_geometries([pcd_object])
            # o3d.visualization.draw_geometries([pcd_background])
            # o3d.visualization.draw_geometries([hull_background_trans])

            o3d.io.write_triangle_mesh("./" + file_name + ".obj", hull_object_trans)
            o3d.io.write_triangle_mesh("./" + file_name + "_background.obj", hull_background_trans)


        finally:
            self.pipeline.stop()
            
        return center
    



if __name__ == '__main__':

    # cmtx, dist = read_camera_parameters()
    camera = Camera()

    output_image = camera.reconstruct()
    # np.set_printoptions(threshold=np.inf)
    # print(output_image)


    

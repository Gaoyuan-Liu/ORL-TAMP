import cv2
import numpy as np
import sys, os, time, math
import pyrealsense2 as rs
import open3d as o3d
import copy

# Plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from scipy.spatial.transform import Rotation
import cv2

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../pddlstream/')
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion
# from examples.pybullet.utils.pybullet_tools.utils import connect, wait_for_user
from examples.pybullet.utils.pybullet_tools.utils import multiply

np.set_printoptions(threshold=np.inf)

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Enable the color stream
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Enable the color stream

        self.align = rs.align(rs.stream.depth)
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
        # self.intrinsics = np.array(self.intrinsics)


        


    def image_flow(self) -> dict: # return a dictionary of aruco id and its center point

        start_time = time.time()

        while True:
            # Wait for the next frame
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            # color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            print(type(depth_frame))

            if not depth_frame:
                continue

            # Convert the color frame to a numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.4), cv2.COLORMAP_JET)
            # depth_image = cv2.resize(depth_image, (64, 48))

            edge_layer = np.zeros((depth_image.shape[0], depth_image.shape[1]))
            for w in range(0, depth_image.shape[0]):
                for h in range(0, depth_image.shape[1]):
                    depth = depth_image[w, h]
                    edge_layer[w, h] = 1 if (depth > 385) and (depth < 410) else 0 

            # print(depth_image)
            # plt.imshow(edge_layer)
            # plt.show()        
            # Display the image with marker detection
            cv2.imshow("depth", edge_layer)
            
            # Break when it's stable:
            if time.time() - start_time > 20 or cv2.waitKey(1) & 0xFF == 27:
                break
            

        # return 0



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

    def reconstruct(self):
        pc = rs.pointcloud()
        points = rs.points()
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
            depth_image_rs_object = copy.deepcopy(depth_image_rs)
            depth_image_rs_with_collision = copy.deepcopy(depth_image_rs)
            color_image_rs = np.asanyarray(aligned_frames.get_color_frame().get_data())

            base_depth = depth_image_rs[int(depth_image_rs.shape[0]/2), int(depth_image_rs.shape[1]/2)]

            for w in range(0, depth_image_rs.shape[0]):
                for h in range(0, depth_image_rs.shape[1]):
                    depth = depth_image_rs[w, h]
                    depth_image_rs_object[w, h] = depth_image_rs[w, h] if (depth < base_depth + 10) and (depth > 0) else 0 
                    depth_image_rs_with_collision[w, h] = depth_image_rs[w, h] if (depth < base_depth + 50) and (depth > 0) else 0


            depth_image = o3d.geometry.Image(depth_image_rs_object)
            depth_image_with_collision = o3d.geometry.Image(depth_image_rs_with_collision)
            color_image = o3d.geometry.Image(color_image_rs)

            
            # Create rgbd image from color and depth images
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image)
            
            rgbd_image_with_collision = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image_with_collision)
            
            
            
            # Create point cloud from rgbd image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, pinhole_camera_intrinsics)
            
            pcd_with_collision = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image_with_collision, pinhole_camera_intrinsics)
            
            
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], 
                            [0, 0, -1, 0], [0, 0, 0, 1]])
            
            pcd_with_collision.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                            [0, 0, -1, 0], [0, 0, 0, 1]])
            
            # pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            pcd_with_collision.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            center = pcd.get_center()

            # Convex hull
            hull, _ = pcd.compute_convex_hull()
            hull_with_collision, _ = pcd_with_collision.compute_convex_hull()
            
            hull_trans = copy.deepcopy(hull).translate(-np.array(center))
            hull_trans.vertices = o3d.utility.Vector3dVector(np.asarray(hull_trans.vertices) * np.array([1, 1, 2]))

            hull_with_collision_trans = copy.deepcopy(hull_with_collision).translate(-np.array(center))
            # hull_with_collision_trans.vertices = o3d.utility.Vector3dVector(np.asarray(hull_with_collision_trans.vertices) * np.array([1, 1, 2]))
            o3d.visualization.draw_geometries([hull_with_collision_trans])
            o3d.io.write_triangle_mesh("./reward_hull.obj", hull_with_collision_trans)

        finally:
            self.pipeline.stop()         


if __name__ == '__main__':

    # cmtx, dist = read_camera_parameters()
    camera = Camera()

    camera.reconstruct()

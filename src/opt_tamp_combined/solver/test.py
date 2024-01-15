import open3d as o3d
import os, sys
import rospy
import pybullet as pb


file_path = os.path.dirname(os.path.realpath(__file__))
# hull = o3d.io.read_triangle_mesh(file_path + '/hull.obj')
# hull_background = o3d.io.read_triangle_mesh(file_path + '/hull_background.obj')

# o3d.visualization.draw_geometries([hull])
# o3d.visualization.draw_geometries([hull_background])
sys.path.insert(0, file_path + '/../../pddlstream/')
from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj
from examples.pybullet.utils.pybullet_tools.franka_primitives import Observe, ObserveObject


rospy.init_node('run_real', anonymous=True)

# try:
#     os.remove('./hull_background.obj')
#     os.remove('./hull.obj')
# except FileNotFoundError:
#     print('File not found')

# observe_numerical = Observe()
# pose_dict = observe_numerical.control(['3']) 
# observe_object = ObserveObject()
# observe_object.reconstruct(pose_dict['3'])


connect(use_gui=True)
obstacle = create_obj(file_path + '/hull_background.obj')
plate = create_obj(file_path + '/hull.obj')

pb.resetBasePositionAndOrientation(obstacle, [1,1,0.5], [0,0,0,1])
pb.resetBasePositionAndOrientation(plate, [1,1,0.5], [0,0,0,1])

input('Press enter to continue: ')
# try:
#     os.remove(file_path + '/hull_background.obj')
#     os.remove(file_path + '/hull.obj')
# except FileNotFoundError:
#     print('File not found')
# print(file_path)
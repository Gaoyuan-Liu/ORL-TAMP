
import os, sys, time, math
import pybullet as pb
import pandas as pd
import numpy as np


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
sys.path.insert(0, file_path + '/../pddlstream/')

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj, multiply, invert

from examples.pybullet.utils.pybullet_tools.panda_utils import get_edge_grasps
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

from common.scn import Scenario, load_ee

def reward_visualization(dish, obstacle):
    obj_pose = pb.getBasePositionAndOrientation(dish)

    # Can be grasped
    obstacles = obstacle
    grasps = get_edge_grasps(dish)
    for grasp in grasps:

        ee = load_ee(pos=(1,-1,1), mode='open')
        grasp_pose = multiply(obj_pose, invert(grasp)) 

        pb.setRealTimeSimulation(0)
        pb.resetBasePositionAndOrientation(ee, grasp_pose[0], grasp_pose[1])
        pb.performCollisionDetection()
        
        collision = False
        for i in obstacles:
            contact_points = pb.getContactPoints(ee, i)
            if len(contact_points) > 0:
                collision = True
                pb.changeVisualShape(ee, 0, rgbaColor=[1,0,0,0.5])
                pb.changeVisualShape(ee, 1, rgbaColor=[1,0,0,0.5])
                pb.changeVisualShape(ee, 2, rgbaColor=[1,0,0,0.5])
                # pb.changeVisualShape(ee, 0, rgbaColor=[1,0,0,1])
                break

        if collision == False:
            print(' Possible Grasp!')
            pb.changeVisualShape(ee, 0, rgbaColor=[0,1,0,0.5])
            pb.changeVisualShape(ee, 1, rgbaColor=[0,1,0,0.5])
            pb.changeVisualShape(ee, 2, rgbaColor=[0,1,0,0.5])

         

def main():

    pb.connect(pb.GUI)

    scn = Scenario()

    mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../camera/'
    pb.setAdditionalSearchPath(mark_dir)
    
    collision = create_obj(mark_dir + 'hull_background.obj', mass=1, color=(1,0,0,0.6))

    dish = create_obj(mark_dir + 'hull.obj', mass=1)

    quat = quaternion_from_euler(0, 0, -math.pi/2)

    pb.resetBasePositionAndOrientation(collision, [1,1,0.5], quat)

    pb.resetBasePositionAndOrientation(dish, [1,1,0.5], quat)

    input('press any key to continue')

    collision = [collision]

    reward_visualization(dish, collision)

    input('press any key to continue')





    

if __name__ == '__main__':
    main()
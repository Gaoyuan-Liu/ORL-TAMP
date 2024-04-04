
import numpy as np
import os, sys, time
from matplotlib import pyplot as plt
import math
import random

# file_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, file_path + '/../utils/')
# sys.path.insert(0, file_path + '/../')



import pybullet as pb
# from utils.pybullet_tools.transformations import quaternion_from_euler

import pybullet_data



# -------
# Ground
# -------
def add_ground():
    data_path = pybullet_data.getDataPath()
    pb.setAdditionalSearchPath(data_path)
    floor = pb.loadURDF("plane.urdf")
    pb.setGravity(0,0,-9.8)



# ------
# Robot
# ------
def add_robot(pose=((0,0,0), (0,0,0,1))):

    flags = pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    orn=[0.0, 0.0, 0.0, 1]#pb.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    # eul = pb.getEulerFromQuaternion([0, 0.5])
    panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/franka_panda/'
    pb.setAdditionalSearchPath(panda_dir)
    robot = pb.loadURDF("panda_modified.urdf", pose[0], pose[1], useFixedBase=True, flags=flags)
    index = 0

    # Create a constraint to keep the fingers centered
    c = pb.createConstraint(robot,
                    9,
                    robot,
                    10,
                    jointType=pb.JOINT_GEAR,
                    jointAxis=[1, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0], 
                    )
    pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
    # Reset the robot
    jointPositions=[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02]
   
    for j in range(pb.getNumJoints(robot)):
        pb.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
        info = pb.getJointInfo(robot, j)
        jointName = info[1]
        jointType = info[2]
        if (jointType == pb.JOINT_PRISMATIC): # 9, 10
            pb.resetJointState(robot, j, jointPositions[index]) 
            index=index+1
        if (jointType == pb.JOINT_REVOLUTE):
            pb.resetJointState(robot, j, jointPositions[index]) 
            index=index+1
    return robot

# ---------
# EE Mobile
# ---------
def add_ee_mobile(pose=((0,0,0), (0,0,0,1)), mode='close'):
    ee_mobile_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/franka_panda/'
    pb.setAdditionalSearchPath(ee_mobile_dir)
    ee_mobile = pb.loadURDF("ee_mobile.urdf", pose[0], pose[1], useFixedBase=True)
    
    # Close fingers
    if mode == 'open':
        jointPositions = [0.04, 0.04]
    elif mode == 'close':
        jointPositions = [0.0, 0.0]
    else:
        raise ValueError('Mode not supported, please choose from open, close.')
    
    pb.resetJointState(ee_mobile, 6, jointPositions[0]) 
    pb.resetJointState(ee_mobile, 7, jointPositions[1])
    return ee_mobile


# ---------
# Bar
# ---------
def add_bar(pose=((0.5,0,0.1), (0,0,0,1))):
    # -----
    # bar
    # -----
    dir = os.path.dirname(os.path.realpath(__file__)) + '/models/bar/'
    bar = pb.loadURDF(dir + 'bar_short.urdf', basePosition=pose[0], baseOrientation=pose[1])
    texUid = pb.loadTexture(dir + "wooden.png")
    pb.changeVisualShape(bar, 0, textureUniqueId=texUid)
    pb.changeVisualShape(bar, -1, textureUniqueId=texUid)
    pb.changeVisualShape(bar, 1, textureUniqueId=texUid)
    return bar


    
# ---------
# Cube
# ---------
def add_cube(pose=((0.5,0,0), (0,0,0,1))):
    halfExtents = np.array([.04, .04, .04])/2
    collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
    visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
    cube = pb.createMultiBody(baseMass = 0.5,
                                    baseCollisionShapeIndex = collision_id,
                                    baseVisualShapeIndex = visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    return cube

# ---------
# Cube
# ---------
def add_cube_random(pose=((0.5,0,0), (0,0,0,1))):
    halfExtents = np.random.uniform(low=(0.04/2, 0.04/2, 0.04/2), high=(0.1/2, 0.1/2, 0.04/2))
    collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
    visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
    cube = pb.createMultiBody(baseMass = 0.1,
                                    baseCollisionShapeIndex = collision_id,
                                    baseVisualShapeIndex = visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    return cube


# -------
# Surface
# -------
def add_surface(pose=((-0.5,0.0,0), (0,0,0,1))):

    surface_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/surface/'
    pb.setAdditionalSearchPath(surface_dir)
    surface = pb.loadURDF('surface.urdf', basePosition=pose[0])
    texUid = pb.loadTexture("wooden_light.png")
    pb.changeVisualShape(surface, -1, textureUniqueId=texUid)
    return surface


# ------
# Cylinder
# ------
def add_cylinder(radius, height, mass=1, color=(1,1,1,1), pose = ((0,0,1), (0,0,0,1))):
    collision_id = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=radius, height=height)    
    visual_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    cylinder = pb.createMultiBody(baseMass = mass,
                                    baseCollisionShapeIndex = collision_id,
                                    baseVisualShapeIndex = visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    return cylinder

# -------
# 
# -------
def add_low_table(pose=((0.75,0,0.075), (0,0,0,1))):
    table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
    pb.setAdditionalSearchPath(table_dir)
    low_table = pb.loadURDF('table_0.3.urdf', basePosition=pose[0])
    texUid = pb.loadTexture("table.png")
    pb.changeVisualShape(low_table, 0, textureUniqueId=texUid)
    return low_table

# -------
#
# -------
def add_low_table_real(pose=((0.75,0,0.18/2), (0,0,0,1))):
    table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
    pb.setAdditionalSearchPath(table_dir)
    halfExtents = np.array([1.0, 0.5, 0.18])/2
    table_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
    table_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1,1,1,1])
    low_table = pb.createMultiBody(baseMass = 0,
                                    baseCollisionShapeIndex = table_collision_id,
                                    baseVisualShapeIndex = table_visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    texUid = pb.loadTexture("table.png")
    pb.changeVisualShape(low_table, -1, textureUniqueId=texUid)
    return low_table

# -------
#
# -------
def add_high_table(pose=((0.5, -0.30-0.17/2, 0.40/2), (0,0,0,1))):
    table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
    pb.setAdditionalSearchPath(table_dir)
    halfExtents = np.array([.4, .17, 0.40])/2
    table_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
    table_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1,1,1,1])
    high_table = pb.createMultiBody(baseMass = 0,
                                    baseCollisionShapeIndex = table_collision_id,
                                    baseVisualShapeIndex = table_visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    
    texUid = pb.loadTexture("table.png")
    pb.changeVisualShape(high_table, -1, textureUniqueId=texUid)
    return high_table

# -------
#
# -------
def add_edge(pose=((0,0,0.075), (0,0,0,1))):
    dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
    edge = pb.loadURDF(dir + 'edge.urdf', basePosition=pose[0])
    texUid = pb.loadTexture(dir + 'table.png')
    pb.changeVisualShape(edge, 0, textureUniqueId=texUid)
    return edge




def add_plate_random(pose=((0.5,0,0.2), (0,0,0,1))):
    if np.random.uniform(low=0.0, high=1.0) > 0.5:
        size = np.random.uniform(low=0.06, high=0.1) # radius
        # size = 0.08
        collision_id = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=size, height=0.03)
        visual_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=size, length=0.02, rgbaColor=[0.6,0.6,0.6,1])
        body_type = 0 # cylinder
    else:
        size = np.random.uniform(low=(0.06, 0.06, 0.015), high=(0.1, 0.1, 0.015)) # half
        collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=size)
        visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=size, rgbaColor=[0.6,0.6,0.6,1])
        body_type = 1 # box
    
    plate = pb.createMultiBody(baseMass = 0.5,
                                    baseCollisionShapeIndex = collision_id,
                                    baseVisualShapeIndex = visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    return plate, body_type

def add_plate(pose=((0.5,0,0.2), (0,0,0,1))):
    
    # size = np.random.uniform(low=0.06, high=0.1) # radius
    size = 0.08
    collision_id = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=size, height=0.03)
    visual_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=size, length=0.02, rgbaColor=[0.6,0.6,0.6,1])
    body_type = 0 # cylinder
    
    
    plate = pb.createMultiBody(baseMass = 0.5,
                                    baseCollisionShapeIndex = collision_id,
                                    baseVisualShapeIndex = visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])
    return plate, body_type


def add_cup(pose=((1.0,0.2,0.4), (0,0,0,1))):
    mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/cup/'
    pb.setAdditionalSearchPath(mark_dir)
    cylinder_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.03]) #pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=0.02, height=0.04)
    cylinder_visual_id = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='cup.obj', meshScale=[], rgbaColor=[0.6,0.6,0.6,1], visualFrameOrientation=[ 0.9999997, 0, 0, 0.0007963 ])
    # cylinder_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05], rgbaColor=[0.6,0.6,0.6,1])
    cup = pb.createMultiBody(baseMass = 0.1,
                                    baseCollisionShapeIndex = cylinder_collision_id,
                                    baseVisualShapeIndex = cylinder_visual_id,
                                    basePosition = pose[0],
                                    baseOrientation = pose[1])

    return cup


def take_a_pic(type='rgb'):
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[1, 1, 1],
        cameraTargetPosition=[0.5, 0, 0.5],
        cameraUpVector=[0, 0, 1])

    projectionMatrix = pb.computeProjectionMatrixFOV(
                    fov= 65,#math.atan(0.5) * (180/math.pi) * 2 #
                    aspect= 1.0,#1.0,
                    nearVal= 0.1,
                    farVal=5)
    
    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                                                width=1000, 
                                                height=1000,
                                                viewMatrix=viewMatrix,
                                                projectionMatrix=projectionMatrix, 
                                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    if type == 'rgb':
        return rgbImg
    elif type == 'depth':
        return depthImg
    elif type == 'seg':
        return segImg
    else:
        raise ValueError('Type not supported, please choose from rgb, depth, seg.')
    

def reset_robot(robot):
    jointPositions=[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02]

    index = 0
    for j in range(pb.getNumJoints(robot)):
        pb.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
        info = pb.getJointInfo(robot, j)
        jointName = info[1]
        jointType = info[2]
        if (jointType == pb.JOINT_PRISMATIC): # 9, 10
            pb.resetJointState(robot, j, jointPositions[index]) 
            index=index+1
        if (jointType == pb.JOINT_REVOLUTE):
            pb.resetJointState(robot, j, jointPositions[index]) 
            index=index+1


#########################################################################################
    
def load_ee(pos=(0,0,0), orn=(0,0,0,1), mode='close'):
    flags = pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/franka_panda/'
    pb.setAdditionalSearchPath(panda_dir)
    ee = pb.loadURDF("ee.urdf", pos, orn, useFixedBase=True, flags=flags)

    if mode == 'open':
        jointPositions = [0.04, 0.04]
    elif mode == 'close':
        jointPositions = [0.0, 0.0]
    else:
        raise ValueError('Mode not supported, please choose from open, close.')
    index = 0
    for j in range(pb.getNumJoints(ee)):
            info = pb.getJointInfo(ee, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == pb.JOINT_PRISMATIC): # 9, 10
                pb.resetJointState(ee, j, jointPositions[index]) 
                index=index+1
    return ee


#########################################################################################

def add_table(size_x, size_y, size_z, mass=0, color=(1,1,1,1), pose = ((0,0,0.5), (0,0,0,1))):
    dir = os.path.dirname(os.path.realpath(__file__)) + "/models/table/"
    # table = pb.loadURDF(dir + "table.urdf", pose[0], pose[1], useFixedBase=True)

    thickness = 0.05

    epsilon = thickness/2

    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[size_x/2, size_y/2, size_z/2])
    visualShapeId = pb.createVisualShapeArray(shapeTypes=[pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX, pb.GEOM_BOX],
                                         halfExtents=[[size_x/2, size_y/2, thickness/2], 
                                                      [thickness/2, thickness/2, size_z/2],
                                                      [thickness/2, thickness/2, size_z/2],
                                                      [thickness/2, thickness/2, size_z/2],
                                                      [thickness/2, thickness/2, size_z/2]],
                                        #  fileNames=["", ""],
                                         visualFramePositions=[
                                                [0, 0, size_z/2-thickness/2],
                                                [size_x/2-0.05, size_y/2-0.05, 0],
                                                [-size_x/2+0.05, size_y/2-0.05, 0],
                                                [-size_x/2+0.05, -size_y/2+0.05, 0],
                                                [size_x/2-0.05, -size_y/2+0.05, 0]
                                                ]

                                         )
    table = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=pose[0], baseOrientation=pose[1])
    
    texUid = pb.loadTexture(dir + "table.png")
    pb.changeVisualShape(table, -1, textureUniqueId=texUid)
    return table



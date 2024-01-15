
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



class Scenario:
    def __init__(self, bullet_client=0) -> None:
        print(' Create a scenario.')
        self.bullet_client = bullet_client

        # -------
        # Ground
        # -------
        data_path = pybullet_data.getDataPath()
        pb.setAdditionalSearchPath(data_path)
        self.floor = pb.loadURDF("plane.urdf", physicsClientId=self.bullet_client)
        pb.setGravity(0,0,-9.8, physicsClientId=self.bullet_client)

    def reset(self):
        self.cubes = []
        self.bars = []
        self.surfaces = []
        self.cups = []


    def add_robot(self, pose=((0,0,0), (0,0,0,1))):
        # pb.setGravity(0,0,-10)

 

        # ------
        # Robot
        # ------
        # self.robot = create_panda()
        # pb.resetBasePositionAndOrientation(self.robot, pose[0], pose[1])

        flags = pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0.0, 0.0, 0.0, 1]#pb.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = pb.getEulerFromQuaternion([0, 0.5])
        panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/franka_panda/'
        pb.setAdditionalSearchPath(panda_dir)
        self.robot = pb.loadURDF("panda_modified.urdf", pose[0], pose[1], useFixedBase=True, flags=flags, physicsClientId=self.bullet_client)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2

        # Create a constraint to keep the fingers centered
        c = pb.createConstraint(self.robot,
                        9,
                        self.robot,
                        10,
                        jointType=pb.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0], 
                        physicsClientId=self.bullet_client)
        pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=self.bullet_client)

        jointPositions=[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02]
    
        for j in range(pb.getNumJoints(self.robot)):
            pb.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            info = pb.getJointInfo(self.robot, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == pb.JOINT_PRISMATIC): # 9, 10
                pb.resetJointState(self.robot, j, jointPositions[index]) 
                index=index+1
            if (jointType == pb.JOINT_REVOLUTE):
                pb.resetJointState(self.robot, j, jointPositions[index]) 
                index=index+1
        return self.robot
    
    def add_ee_mobile(self, pose=((0,0,0), (0,0,0,1)), mode='close'):
        ee_mobile_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/franka_panda/'
        pb.setAdditionalSearchPath(ee_mobile_dir)
        self.ee_mobile = pb.loadURDF("ee_mobile.urdf", pose[0], pose[1], useFixedBase=True, physicsClientId=self.bullet_client)
        
        # Close fingers
        if mode == 'open':
            jointPositions = [0.04, 0.04]
        elif mode == 'close':
            jointPositions = [0.0, 0.0]
        else:
            raise ValueError('Mode not supported, please choose from open, close.')
       
        pb.resetJointState(self.ee_mobile, 6, jointPositions[0]) 
        pb.resetJointState(self.ee_mobile, 7, jointPositions[1])

        # for j in range(pb.getNumJoints(self.ee_mobile)):
        #     info = pb.getJointInfo(self.ee_mobile, j)
        #     print(info)


        return self.ee_mobile

    

    def add_bar(self, pose=((0.5,0,0.1), (0,0,0,1))):
        # -----
        # bar
        # -----
        bar_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/bar/'
        pb.setAdditionalSearchPath(bar_dir)
        self.bar = pb.loadURDF('bar.urdf', basePosition=pose[0], baseOrientation=pose[1])
        texUid = pb.loadTexture("wooden.png")
        pb.changeVisualShape(self.bar, 0, textureUniqueId=texUid)
        pb.changeVisualShape(self.bar, -1, textureUniqueId=texUid)
        pb.changeVisualShape(self.bar, 1, textureUniqueId=texUid)
        self.bars.append(self.bar)
        return self.bar
    

        

    def add_cube(self, pose=((0.5,0,0), (0,0,0,1))):
        halfExtents = np.array([.04, .04, .04])/2
        collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
        visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
        self.cube = pb.createMultiBody(baseMass = 0.5,
                                        baseCollisionShapeIndex = collision_id,
                                        baseVisualShapeIndex = visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        return self.cube
    
    def add_cube_random(self, pose=((0.5,0,0), (0,0,0,1))):
        halfExtents = np.random.uniform(low=(0.04/2, 0.04/2, 0.04/2), high=(0.1/2, 0.1/2, 0.04/2))
        collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
        visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
        self.cube = pb.createMultiBody(baseMass = 0.1,
                                        baseCollisionShapeIndex = collision_id,
                                        baseVisualShapeIndex = visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        return self.cube

    def add_surface(self, pose=((-0.5,0.0,0), (0,0,0,1))):
        # -------
        # Surface
        # -------
        surface_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/surface/'
        pb.setAdditionalSearchPath(surface_dir)
        self.surface = pb.loadURDF('surface.urdf', basePosition=pose[0])
        self.surfaces.append(self.surface)
        texUid = pb.loadTexture("wooden_light.png")
        pb.changeVisualShape(self.surface, -1, textureUniqueId=texUid)
        return self.surface
    
    def add_low_table(self, pose=((0.75,0,0.075), (0,0,0,1))):
        # -------
        # Surface
        # -------
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
        pb.setAdditionalSearchPath(table_dir)
        self.low_table = pb.loadURDF('table_0.3.urdf', basePosition=pose[0])
        self.surfaces.append(self.low_table)
        texUid = pb.loadTexture("table.png")
        pb.changeVisualShape(self.low_table, 0, textureUniqueId=texUid)
        self.surfaces.append(self.low_table)
        return self.low_table
    
    def add_low_table_real(self, pose=((0.75,0,0.18/2), (0,0,0,1))):
        # -------
        # Surface
        # -------
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
        pb.setAdditionalSearchPath(table_dir)
        # self.low_table = pb.loadURDF('table_0.3.urdf', basePosition=pose[0])
        halfExtents = np.array([1.0, 0.5, 0.18])/2
        table_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
        table_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1,1,1,1])
        self.low_table = pb.createMultiBody(baseMass = 0,
                                        baseCollisionShapeIndex = table_collision_id,
                                        baseVisualShapeIndex = table_visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        self.surfaces.append(self.low_table)
        texUid = pb.loadTexture("table.png")
        pb.changeVisualShape(self.low_table, -1, textureUniqueId=texUid)
        self.surfaces.append(self.low_table)
        return self.low_table
    
    def add_high_table(self, pose=((0.5, -0.30-0.17/2, 0.40/2), (0,0,0,1))):
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
        pb.setAdditionalSearchPath(table_dir)
        halfExtents = np.array([.4, .17, 0.40])/2
        table_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents)
        table_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[1,1,1,1])
        self.high_table = pb.createMultiBody(baseMass = 0,
                                        baseCollisionShapeIndex = table_collision_id,
                                        baseVisualShapeIndex = table_visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        
        texUid = pb.loadTexture("table.png")
        pb.changeVisualShape(self.high_table, -1, textureUniqueId=texUid)
        self.surfaces.append(self.high_table)
        return self.high_table
    

    def add_edge(self, pose=((0,0,0.075), (0,0,0,1))):
        dir = os.path.dirname(os.path.realpath(__file__)) + '/models/table/'
        self.low_table = pb.loadURDF(dir + 'edge.urdf', basePosition=pose[0])
        self.surfaces.append(self.low_table)
        texUid = pb.loadTexture(dir + 'table.png')
        pb.changeVisualShape(self.low_table, 0, textureUniqueId=texUid)
        return self.low_table
    

    def add_cup_mark(self, pose=((0.5,0,0), (0,0,0,1)), color='green'):
        # ------------
        # Cubes Shadow
        # ------------
        cube_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/cup/'
        pb.setAdditionalSearchPath(cube_dir)
        if color == 'green':
            self.cup_mark = pb.loadURDF('cup_shadow_green.urdf', basePosition=pose[0], baseOrientation=pose[1])
        elif color == 'red':
            self.cup_mark = pb.loadURDF('cup_shadow_red.urdf', basePosition=pose[0], baseOrientation=pose[1])
        return self.cup_mark

    def add_goal_mark(self, pose=((-0.5,0,0.1), (0,0,0,1)), color='green'):
        # ---------
        # Goal Mark
        # ---------
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/goal_mark/'
        pb.setAdditionalSearchPath(mark_dir)
        if color == 'green':
            self.goal_mark = pb.loadURDF('goal_mark_green.urdf', basePosition=pose[0])
        elif color == 'red':
            self.goal_mark = pb.loadURDF('goal_mark_red.urdf', basePosition=pose[0])
        return self.goal_mark
    

    
    def add_hook_mark(self, pose=((0.5,0.2,0.0001), (0,0,0,1)), color='green'):
        # ---------
        # Goal Mark
        # ---------
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/hook_mark/'
        pb.setAdditionalSearchPath(mark_dir)
        if color == 'green':
            self.hook_mark = pb.loadURDF('hook_mark_green.urdf', basePosition=pose[0], baseOrientation=pose[1])
        elif color == 'red':
            self.hook_mark = pb.loadURDF('hook_mark_red.urdf', basePosition=pose[0], baseOrientation=pose[1])
        else:
            raise ValueError('Color not supported')
        return self.hook_mark
    
   
    
    def add_plate_random(self, pose=((0.5,0,0.2), (0,0,0,1))):
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
        
        self.plate = pb.createMultiBody(baseMass = 0.5,
                                        baseCollisionShapeIndex = collision_id,
                                        baseVisualShapeIndex = visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        return self.plate, body_type
    
    def add_plate(self, pose=((0.5,0,0.2), (0,0,0,1))):
       
        # size = np.random.uniform(low=0.06, high=0.1) # radius
        size = 0.08
        collision_id = pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=size, height=0.03)
        visual_id = pb.createVisualShape(shapeType=pb.GEOM_CYLINDER, radius=size, length=0.02, rgbaColor=[0.6,0.6,0.6,1])
        body_type = 0 # cylinder
       
        
        self.plate = pb.createMultiBody(baseMass = 0.5,
                                        baseCollisionShapeIndex = collision_id,
                                        baseVisualShapeIndex = visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        return self.plate, body_type

    
    def add_cup(self, pose=((1.0,0.2,0.4), (0,0,0,1))):
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/models/cup/'
        pb.setAdditionalSearchPath(mark_dir)
        cylinder_collision_id = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05]) #pb.createCollisionShape(shapeType=pb.GEOM_CYLINDER, radius=0.02, height=0.04)
        # cylinder_visual_id = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName='cup.obj', meshScale=[], rgbaColor=[0.6,0.6,0.6,1], visualFrameOrientation=[ 0.9999997, 0, 0, 0.0007963 ])
        cylinder_visual_id = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.02, 0.02, 0.05], rgbaColor=[0.6,0.6,0.6,1])
        self.cup = pb.createMultiBody(baseMass = 0.1,
                                        baseCollisionShapeIndex = cylinder_collision_id,
                                        baseVisualShapeIndex = cylinder_visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        # self.cup = pb.loadURDF('cup.urdf', basePosition=pose[0], baseOrientation=pose[1])
        self.cups.append(self.cup)
        return self.cup
    

    def take_a_pic(self, type='rgb'):
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
        

    def reset_robot(self, robot=None):
        jointPositions=[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02]
        if robot == None:
            robot = self.robot

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
            # pb.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            info = pb.getJointInfo(ee, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == pb.JOINT_PRISMATIC): # 9, 10
                pb.resetJointState(ee, j, jointPositions[index]) 
                index=index+1
    return ee



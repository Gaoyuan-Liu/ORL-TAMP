
import numpy as np
import os, sys, time
from matplotlib import pyplot as plt
import math
import random

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../utils/')
sys.path.insert(0, file_path + '/../')


from pybullet_tools.utils import connect, get_pose, is_placement, disconnect, \
    get_joint_positions, HideOutput, LockRenderer, wait_for_user, add_data_path


import pybullet as p
# from utils.pybullet_tools.transformations import quaternion_from_euler
import cv2



class Scenario:
    def __init__(self, bullet_client=p) -> None:
        print('Create a scenario.')
        self.bullet_client = bullet_client

        # -------
        # Ground
        # -------
        add_data_path()
        self.floor = p.loadURDF("plane.urdf")
        # ground_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/ground/'
        # p.setAdditionalSearchPath(ground_dir)
        # self.floor = p.loadURDF('ground.urdf', basePosition=[0,0,-5])
        p.setGravity(0,0,-9.8)

    def reset(self):
        self.cubes = []
        self.hooks = []
        self.surfaces = []


    def add_robot(self, pose=((0,0,0.2), (0,0,0,1))):
        # p.setGravity(0,0,-10)
        data_path = add_data_path()
 
        # panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/franka_description/robots/'
        # p.setAdditionalSearchPath(panda_dir)
        # self.panda = p.loadURDF('panda_arm_hand.urdf', basePosition=pose[0])

        # ------
        # Robot
        # ------
        # self.panda = create_panda()
        # p.resetBasePositionAndOrientation(self.panda, pose[0], pose[1])

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0.0, 0.0, 0.0, 1]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = p.getEulerFromQuaternion([0, 0.5])
        panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/franka_panda/'
        p.setAdditionalSearchPath(panda_dir)
        self.panda = p.loadURDF("panda_modified.urdf", pose[0], pose[1], useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2

        # Create a constraint to keep the fingers centered
        c = p.createConstraint(self.panda,
                        9,
                        self.panda,
                        10,
                        jointType=p.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        jointPositions=[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02]
    
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC): # 9, 10
                p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        return self.panda

    

    def add_hook(self, pose=((0.5,0,0.1), (0,0,0,1))):
        # -----
        # Hook
        # -----
        hook_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/hook/'
        p.setAdditionalSearchPath(hook_dir)
        self.hook = p.loadURDF('hook.urdf', basePosition=pose[0], baseOrientation=pose[1])
        texUid = p.loadTexture("wooden.png")
        p.changeVisualShape(self.hook, 0, textureUniqueId=texUid)
        p.changeVisualShape(self.hook, -1, textureUniqueId=texUid)
        p.changeVisualShape(self.hook, 1, textureUniqueId=texUid)
        self.hooks.append(self.hook)
        return self.hook

    def add_bar(self, pose=((0.5,0,0.1), (0,0,0,1))):
        # -----
        # Hook
        # -----
        hook_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/hook/'
        p.setAdditionalSearchPath(hook_dir)
        self.hook = p.loadURDF('bar.urdf', basePosition=pose[0], baseOrientation=pose[1])
        texUid = p.loadTexture("wooden.png")
        p.changeVisualShape(self.hook, 0, textureUniqueId=texUid)
        p.changeVisualShape(self.hook, -1, textureUniqueId=texUid)
        p.changeVisualShape(self.hook, 1, textureUniqueId=texUid)
        self.hooks.append(self.hook)
        return self.hook
        

    def add_cube(self, pose=((0.5,0,0), (0,0,0,1))):
        # ------
        # Cubes
        # ------
        cube_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/cube/'
        p.setAdditionalSearchPath(cube_dir)
        self.cube = p.loadURDF('cube.urdf', basePosition=pose[0])
        self.cubes.append(self.cube)
        return self.cube

    def add_surface(self, pose=((-0.5,0.0,0), (0,0,0,1))):
        # -------
        # Surface
        # -------
        surface_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/surface/'
        p.setAdditionalSearchPath(surface_dir)
        self.surface = p.loadURDF('surface.urdf', basePosition=pose[0])
        self.surfaces.append(self.surface)
        texUid = p.loadTexture("wooden_light.png")
        p.changeVisualShape(self.surface, -1, textureUniqueId=texUid)
        return self.surface
    
    def add_low_table(self, pose=((0.5,0,0.15), (0,0,0,1))):
        # -------
        # Surface
        # -------
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/table/'
        p.setAdditionalSearchPath(table_dir)
        self.low_table = p.loadURDF('table_0.3.urdf', basePosition=pose[0])
        self.surfaces.append(self.low_table)
        # texUid = p.loadTexture("wooden.png")
        # p.changeVisualShape(self.table, 0, textureUniqueId=texUid)

        # table_collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        # table_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.6,0.6,0.6,1])
        # self.table = p.createMultiBody(baseMass = 0,
        #                                 baseCollisionShapeIndex = table_collision_id,
        #                                 baseVisualShapeIndex = table_visual_id,
        #                                 basePosition = pose[0],
        #                                 baseOrientation = pose[1])
        texUid = p.loadTexture("table.png")
        p.changeVisualShape(self.low_table, 0, textureUniqueId=texUid)
        self.surfaces.append(self.low_table)
        return self.low_table
    
    def add_high_table(self, pose=((0.6,-0.5,0), (0,0,0,1))):
        # -------
        # Surface
        # -------
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/table/'
        p.setAdditionalSearchPath(table_dir)
        self.high_table = p.loadURDF('table_0.6.urdf', basePosition=pose[0])
        self.surfaces.append(self.high_table)
        # texUid = p.loadTexture("wooden.png")
        # p.changeVisualShape(self.table, 0, textureUniqueId=texUid)

        # table_collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        # table_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.6,0.6,0.6,1])
        # self.table = p.createMultiBody(baseMass = 0,
        #                                 baseCollisionShapeIndex = table_collision_id,
        #                                 baseVisualShapeIndex = table_visual_id,
        #                                 basePosition = pose[0],
        #                                 baseOrientation = pose[1])
        texUid = p.loadTexture("table.png")
        p.changeVisualShape(self.high_table, 0, textureUniqueId=texUid)
        self.surfaces.append(self.high_table)
        return self.high_table
    
    def add_low_long_table(self, pose=((0.75,0,0), (0,0,0,1))):
        # -------
        # Surface
        # -------
        table_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/table/'
        p.setAdditionalSearchPath(table_dir)
        self.low_table = p.loadURDF('table_0.3_1.25.urdf', basePosition=pose[0])
        self.surfaces.append(self.low_table)
        # texUid = p.loadTexture("wooden.png")
        # p.changeVisualShape(self.table, 0, textureUniqueId=texUid)

        # table_collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        # table_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.6,0.6,0.6,1])
        # self.table = p.createMultiBody(baseMass = 0,
        #                                 baseCollisionShapeIndex = table_collision_id,
        #                                 baseVisualShapeIndex = table_visual_id,
        #                                 basePosition = pose[0],
        #                                 baseOrientation = pose[1])
        texUid = p.loadTexture("table.png")
        p.changeVisualShape(self.low_table, 0, textureUniqueId=texUid)
        self.surfaces.append(self.low_table)
        return self.low_table
    

    def add_cube_mark(self, pose=((0.5,0,0), (0,0,0,1)), color='green'):
        # ------------
        # Cubes Shadow
        # ------------
        cube_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/cube_mark/'
        p.setAdditionalSearchPath(cube_dir)
        if color == 'green':
            self.cube_mark = p.loadURDF('cube_shadow_green.urdf', basePosition=pose[0], baseOrientation=pose[1])
        elif color == 'red':
            self.cube_mark = p.loadURDF('cube_shadow_red.urdf', basePosition=pose[0], baseOrientation=pose[1])
        return self.cube_mark

    def add_goal_mark(self, pose=((-0.5,0,0.1), (0,0,0,1)), color='blue'):
        # ---------
        # Goal Mark
        # ---------
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/goal_mark/'
        p.setAdditionalSearchPath(mark_dir)
        if color == 'blue':
            self.goal_mark = p.loadURDF('goal_mark_blue.urdf', basePosition=pose[0])
        elif color == 'red':
            self.goal_mark = p.loadURDF('goal_mark_red.urdf', basePosition=pose[0])
        return self.goal_mark
    
    def add_goal_surface(self, pose=((-0.5,0,0.001), (0,0,0,1))):
        # ---------
        # Goal Mark
        # ---------
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/goal_mark/'
        p.setAdditionalSearchPath(mark_dir)
       
        self.goal_surface = p.loadURDF('goal_mark.urdf', basePosition=pose[0])
        
        return self.goal_surface
    
    
    def add_hook_mark(self, pose=((0.5,0.2,0.0001), (0,0,0,1)), color='green'):
        # ---------
        # Goal Mark
        # ---------
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/hook_mark/'
        p.setAdditionalSearchPath(mark_dir)
        if color == 'green':
            self.hook_mark = p.loadURDF('hook_mark_green.urdf', basePosition=pose[0], baseOrientation=pose[1])
        elif color == 'red':
            self.hook_mark = p.loadURDF('hook_mark_red.urdf', basePosition=pose[0], baseOrientation=pose[1])
        else:
            raise ValueError('Color not supported')
        return self.hook_mark
    
    def add_dish(self, pose=((0.5,0,0.35), (0,0,0,1))):
        # mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/dish/'
        # p.setAdditionalSearchPath(mark_dir)
        # self.dish = p.loadURDF('dish.urdf', basePosition=pose[0], baseOrientation=pose[1])
        cylinder_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.08, height=0.02)
        cylinder_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.08, length=0.02, rgbaColor=[0.6,0.6,0.6,1])

        self.dish = p.createMultiBody(baseMass = 0.5,
                                        baseCollisionShapeIndex = cylinder_collision_id,
                                        baseVisualShapeIndex = cylinder_visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        return self.dish
    
    def add_cup(self, pose=((1.0,0,0.4), (0,0,0,1))):
        mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/cup/'
        p.setAdditionalSearchPath(mark_dir)
        cylinder_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.08, height=0.07)
        cylinder_visual_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName='cup.obj', meshScale=[], rgbaColor=[0.6,0.6,0.6,1], visualFrameOrientation=[ 0.9999997, 0, 0, 0.0007963 ])

        self.cup = p.createMultiBody(baseMass = 0.5,
                                        baseCollisionShapeIndex = cylinder_collision_id,
                                        baseVisualShapeIndex = cylinder_visual_id,
                                        basePosition = pose[0],
                                        baseOrientation = pose[1])
        # self.cup = p.loadURDF('cup.urdf', basePosition=pose[0], baseOrientation=pose[1])
        return self.cup
    

    def take_a_pic(self, type='rgb'):
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[1, 1, 1],
            cameraTargetPosition=[0.5, 0, 0.5],
            cameraUpVector=[0, 0, 1])
    
        projectionMatrix = p.computeProjectionMatrixFOV(
                        fov= 65,#math.atan(0.5) * (180/math.pi) * 2 #
                        aspect= 1.0,#1.0,
                        nearVal= 0.1,
                        farVal=5)
        
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                                                    width=1000, 
                                                    height=1000,
                                                    viewMatrix=viewMatrix,
                                                    projectionMatrix=projectionMatrix, 
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        if type == 'rgb':
            return rgbImg
        elif type == 'depth':
            return depthImg
        elif type == 'seg':
            return segImg
        else:
            raise ValueError('Type not supported, please choose from rgb, depth, seg.')
        


#########################################################################################
    
def load_ee(pos=(0,0,0), orn=(0,0,0,1)):
    flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    panda_dir = os.path.dirname(os.path.realpath(__file__)) + '/../utils/models/franka_panda/'
    p.setAdditionalSearchPath(panda_dir)
    ee = p.loadURDF("ee.urdf", pos, orn, useFixedBase=True, flags=flags)
    return ee



if __name__ == '__main__':
    # from utils.pybullet_tools.utils import connect, wait_for_user

    connect(use_gui=True)

    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


    scn = Scenario()
    
    scn.reset()

    # Fixed
    scn.add_low_long_table()
    scn.add_high_table()
    scn.add_robot()
    
    #############################
    # Before
    # scn.add_dish(((0.5, 0.0, 0.33), (0,0,0,1)))
    # scn.add_cup(((0.9, 0.1, 0.33), (0,0,0,1)))
    # quat_bar = p.getQuaternionFromEuler((0,0,0.2))
    # scn.add_bar(((0.5, -0.1, 0.33), quat_bar))

    #############################
    # After
    # scn.add_dish(((0.6, -0.4, 0.7), (0,0,0,1)))
    # scn.add_cup(((0.6, -0.4, 0.7), (0,0,0,1)))
    # scn.add_bar(((0.5, 0.1, 0.33), (0,0,0,1)))

    #############################
    # Hook
    # scn.add_dish(((0.5, 0.0, 0.33), (0,0,0,1)))
    # scn.add_cup(((0.7, 0.1, 0.33), (0,0,0,1)))
    # quat_bar = p.getQuaternionFromEuler((0,0,1))
    # scn.add_bar(((0.7, -0.05, 0.33), quat_bar))
    # p.setRealTimeSimulation(1)
    # time.sleep(1)
    # p.setRealTimeSimulation(0)
    
    # # file_path = os.path.dirname(os.path.realpath(__file__))
    # # sys.path.insert(0, file_path)
    # from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path + '/../pddlstream/')
    # from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

    
    # info = p.getBasePositionAndOrientation(scn.hook)
    # ee_pos1 = list(info[0])
    # ee_pos1[-1] = ee_pos1[-1] + 0.1
    # hook_orn = info[1]
    # hook_rpy = euler_from_quaternion(hook_orn)
    # ee_yaw1 = (hook_rpy[-1] - math.pi/4) % (np.sign(hook_rpy[-1] - math.pi/4)*(math.pi))
    # ee_orn1 = quaternion_from_euler(math.pi, 0, ee_yaw1)
    # approach_pose = (ee_pos1, ee_orn1)
    # grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose) # Out put is 7 dof, and it automatically set the joint angles

    #############################
    # Push

    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path)
    # from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path + '/../pddlstream/')
    # from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion


    # scn.add_dish(((0.5, 0.0, 0.33), (0,0,0,1)))
    # p.setRealTimeSimulation(1)
    # time.sleep(1)
    # p.setRealTimeSimulation(0)

    # ee_pos = [0.5, -0.1, 0.43]
    # ee_orn = quaternion_from_euler(math.pi, 0, math.pi/2)
    # approach_pose = (ee_pos, ee_orn)
    # grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose)


    #############################
    # Push Pick 
    
    # from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path + '/../pddlstream/')
    # from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

    # scn.add_dish(((0.4, 0.23, 0.31), (0,0,0,1)))
    # p.setRealTimeSimulation(1)
    # time.sleep(1)
    # p.setRealTimeSimulation(0)

    # ee_pos = [0.4, 0.4, 0.3]
    # ee_orn = quaternion_from_euler(math.pi*0.5, math.pi, 0, axes='sxyz')
    # approach_pose = (ee_pos, ee_orn)
    # grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose)

    #############################
    # Push Place
    # from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path + '/../pddlstream/')
    # from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

    # scn.add_dish(((0.4, -0.26, 0.61), (0,0,0,1)))
    # p.setRealTimeSimulation(1)
    # time.sleep(1)
    # p.setRealTimeSimulation(0)

    # ee_pos = [0.4, -0.1, 0.6]
    # ee_orn = quaternion_from_euler(math.pi*0.5, math.pi, 0, axes='sxyz')
    # approach_pose = (ee_pos, ee_orn)
    # grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose)

    ###############################
    # Hook pick
    # from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sys.path.insert(0, file_path + '/../pddlstream/')
    # from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

    # # scn.add_dish(((0.5, 0.0, 0.33), (0,0,0,1)))
    # scn.add_cup(((0.5, 0.1, 0.33), (0,0,0,1)))
    # quat_bar = p.getQuaternionFromEuler((0,0,1))
    # scn.add_bar(((0.6, -0.05, 0.33), quat_bar))
    # p.setRealTimeSimulation(1)
    # time.sleep(1)
    # p.setRealTimeSimulation(0)

    # ee_pos = [0.5, 0.1, 0.45]
    # ee_orn = quaternion_from_euler(math.pi, 0, 0, axes='sxyz')
    # approach_pose = (ee_pos, ee_orn)
    # grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose)


    ###############################
    # Hook place
    from pybullet_tools.ikfast.franka_panda.ik import panda_inverse_kinematics
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path + '/../pddlstream/')
    from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

    # scn.add_dish(((0.5, 0.0, 0.33), (0,0,0,1)))
    scn.add_cup(((0.4, -0.28, 0.65), (0,0,0,1)))
    scn.add_dish(((0.4, -0.28, 0.61), (0,0,0,1)))
    quat_bar = p.getQuaternionFromEuler((0,0,1))
    scn.add_bar(((0.6, -0.05, 0.33), quat_bar))
    p.setRealTimeSimulation(1)
    time.sleep(1)
    p.setRealTimeSimulation(0)

    ee_pos = [0.4, -0.28, 0.75]
    ee_orn = quaternion_from_euler(math.pi, 0, 0, axes='sxyz')
    approach_pose = (ee_pos, ee_orn)
    grasp_conf = panda_inverse_kinematics(scn.panda, scn.panda, approach_pose)





    

    img = scn.take_a_pic()
    cv2.imwrite('./1.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    

    wait_for_user()
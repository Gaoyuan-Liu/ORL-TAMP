
import sys, os, time
from control import Control
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
from utils.scn import add_ground, add_robot, add_cylinder, add_bar, add_cup, add_table
from utils.pybullet_tools.utils import connect, wait_for_user
from utils.pybullet_tools.panda_utils import get_edge_grasps, get_top_grasps
from utils.pybullet_tools.utils import multiply, invert
import pybullet as pb
import math

if __name__ == '__main__':
    connect(use_gui=True)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

    #################################################
    # Scenario 
    scn = {}

    ground = add_ground()
    scn['ground'] = ground

    robot = add_robot(pose=((0,0,0), (0,0,0,1)))
    scn['robot'] = robot

    table_low = add_table(1, 0.5, 0.15, pose=((0.75, 0, 0.075),(0,0,0,1))) #add_low_table()
    scn['low_table'] = table_low

    table_high = add_table(0.5, 0.5, 0.5, pose=((0.5, -0.5, 0.25),(0,0,0,1))) 
    scn['high_table'] = table_high

    plate = add_cylinder(0.08, 0.03, pose=((0.45, 0.05, 0.15+0.015),(0,0,0,1)), color=(0.6,0.6,0.6,1))
    scn['plate'] = plate
    
    orn = pb.getQuaternionFromEuler([0., 0., 0.1])
    bar = add_bar(((0.6, -0.1, 0.33), orn))
    scn['bar'] = bar

    cup = add_cup(((0.9, 0.1, 0.33), (0,0,0,1)))
    scn['cup'] = cup

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)

    control = Control()

    #################################################
    # Figure 1
    input('\n Figure 1 \n')


    #################################################
    # Figure 2
    # pb.changeVisualShape(scn['plate'], -1, rgbaColor=[0.6,0.6,0.6,1])
    control.finger_close(robot=scn['robot'])
    # control.finger_open(robot=scn['robot'])
    orn = pb.getQuaternionFromEuler([math.pi, 0., math.pi/2])
    control.set_cartesian_space(robot=scn['robot'], eePose=((0.45, -0.04, 0.15+0.12), orn))
    input('\n Figure 2 \n')

    #################################################
    # Figure 3
    pb.resetBasePositionAndOrientation(scn['plate'], [0.45, 0.24, 0.15+0.015], [0,0,0,1])
    obj_pose = pb.getBasePositionAndOrientation(scn['plate'])
    grasps = get_edge_grasps(scn['plate'], body_type = 0)
    for grasp in grasps:
        grasp_pose = multiply(obj_pose, invert(grasp)) 
        control.set_cartesian_space(robot=scn['robot'], eePose=(grasp_pose[0], grasp_pose[1]))
        input(' Next grasp.')
    input('\n Figure 3 \n')

    
    #################################################
    # Figure 4
    pb.resetBasePositionAndOrientation(scn['plate'], [0.45, -0.28, 0.5+0.015], [0,0,0,1])
    obj_pose = pb.getBasePositionAndOrientation(scn['plate'])
    grasps = get_edge_grasps(scn['plate'], body_type = 0)
    for grasp in grasps:
        grasp_pose = multiply(obj_pose, invert(grasp)) 
        control.set_cartesian_space(robot=scn['robot'], eePose=(grasp_pose[0], grasp_pose[1]))
        input(' Next grasp.')
    input('\n Figure 4 \n')

    #################################################
    # Figure 5 (manipulate the bar)
    pb.resetBasePositionAndOrientation(scn['cup'], [0.65, 0.1, 0.18], [0,0,0,1])
    orn = pb.getQuaternionFromEuler([0., 0., math.pi/4])
    pb.resetBasePositionAndOrientation(scn['bar'], [0.6, -0.1, 0.15 + 0.02], orn)
    

    obj_pose = pb.getBasePositionAndOrientation(scn['bar'])
    grasps = get_top_grasps(scn['bar'])
    for grasp in grasps:
        grasp_pose = multiply(obj_pose, invert(grasp)) 
        control.set_cartesian_space(robot=scn['robot'], eePose=(grasp_pose[0], grasp_pose[1]))
        input(' Next grasp.')
    input('\n Figure 5 \n')

    #################################################
    # Figure 6 (grasp the cup)
    pb.resetBasePositionAndOrientation(scn['cup'], [0.65, 0.1, 0.18], [0,0,0,1])
    orn = pb.getQuaternionFromEuler([0., 0., math.pi/4])
    pb.resetBasePositionAndOrientation(scn['bar'], [0.6, -0.1, 0.15 + 0.02], orn)
    

    obj_pose = pb.getBasePositionAndOrientation(scn['cup'])
    grasps = get_top_grasps(scn['cup'])
    for grasp in grasps:
        grasp_pose = multiply(obj_pose, invert(grasp)) 
        control.set_cartesian_space(robot=scn['robot'], eePose=(grasp_pose[0], grasp_pose[1]), fingers='open')
        input(' Next grasp.')
    input('\n Figure 6 \n')

    #################################################
    # Figure 7 (place the cup)
    pb.resetBasePositionAndOrientation(scn['cup'], [0.45, -0.28, 0.5+0.015+0.06], [0,0,0,1])
    

    obj_pose = pb.getBasePositionAndOrientation(scn['cup'])
    grasps = get_top_grasps(scn['cup'])
    for grasp in grasps:
        grasp_pose = multiply(obj_pose, invert(grasp)) 
        control.set_cartesian_space(robot=scn['robot'], eePose=(grasp_pose[0], grasp_pose[1]), fingers='open')
        input(' Next grasp.')
    input('\n Figure 7 \n')

    #################################################
    # Figure 8 (reset)
    pb.resetBasePositionAndOrientation(scn['cup'], [0.45, -0.27, 0.5+0.015+0.05], [0,0,0,1])
    control.set_joint_space(robot=scn['robot'], jointPoses = [0, 0, 0, -1.5, 0, 1.5, 0])
    input('\n Figure 8 \n')




import os, sys, time, math
import pybullet as pb
import pandas as pd
import numpy as np
import rospy
import pickle

from opt_solver import Solver
from execute_real import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
sys.path.insert(0, file_path + '/../../pddlstream/')

from pddlstream.language.constants import Equal, And

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj
from examples.pybullet.utils.pybullet_tools.franka_primitives import Observe, ObserveObject
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion


def main():

    rospy.init_node('run_real', anonymous=True)
    connect(use_gui=True)

    #################################################
    # Planning scene
    scn = reset()
    # S2R
    s2r = {scn.plate:'3', scn.bar:'1', scn.cup:'2'}
    input('\n Scene is ready. Press Enter to plan. ')


    #################################################
    
    solver = Solver(scn.robot, movable=[scn.plate, scn.cup], tools=[scn.bar], surfaces=[scn.low_table, scn.high_table, scn.plate],
                    top_grasps=[scn.cup], edge_grasps=[scn.plate])
    
    executor = Executor(scn, s2r)

    #################################################
    # Define the goal
    # goal = And(('on', scn.plate, table_high), ('on', cup, scn.plate))
    goal = ('on', scn.cup, scn.high_table)
    # goal = ('on', scn.plate, scn.high_table) 
    
    #################################################
    # Solve the problem
    # plan, cost, evaluations = solver.solve(goal)

    solver.problem_formation()
    plan = pickle.load(open("./plan_retrieve.pkl", "rb", -1))
    # plan = pickle.load(open("./plan_edgepush.pkl", "rb", -1))

    if plan is None:
        print('Planning failed')
        return  
    # else:
        # with open('./plan_retrieve.pkl', 'wb') as file:
        #     pickle.dump(plan, file)

    #################################################
    # Execute the plan
    input('\n Scene is ready. Press Enter to execute. ')
    executor.execute(solver.problem, plan)


def reset():
    scn = Scenario()
    scn.reset()

    # Add
    scn.add_robot(pose=((0,0,0), (0,0,0,1)))
    scn.add_low_table_real()
    scn.add_high_table()
    scn.add_bar(((0.6, -0.1, 0.33), (0,0,0,1)))
    scn.add_cup(((0.8, 0.1, 0.24), (0,0,0,1)))


    observe = Observe()
    # pose_dict = observe.control(expected_ids=['0', '1', '3'])
    # pose_dict = observe.control(expected_ids=['0', '1'])
    pose_dict = observe.control(expected_ids=['1', '2'])

    # plate
    # plate_pose = pose_dict['3']
    # observe_object = ObserveObject()
    # observe_object.reconstruct(plate_pose)
    scn.plate = create_obj('./hull.obj', mass=1)
    # # scn.add_plate(pose=((0.5, 0.15, 0.17), (0,0,0,1)))
    quat = quaternion_from_euler(0, 0, -math.pi/2)
    # pb.resetBasePositionAndOrientation(scn.plate, np.array(plate_pose[0])+ np.array([0,-0.05,0]), quat)
    pb.resetBasePositionAndOrientation(scn.plate, np.array([10,10,10])+ np.array([0,-0.05,0]), quat)

    # cup
    cup_pose = pose_dict['2']
    pb.resetBasePositionAndOrientation(scn.cup, np.array(cup_pose[0])+ np.array([0,0,0.]), cup_pose[1])

    # Bar
    bar_pose = pose_dict['1']
    pb.resetBasePositionAndOrientation(scn.bar, np.array(bar_pose[0])+ np.array([0,0,0]), bar_pose[1])


    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    # Print
    print(' \n====== Scenario ====== ')
    print(' Robot: ', scn.robot)
    print(' Low Table: ', scn.low_table)
    print(' High Table: ', scn.high_table)
    # print(' Plate: ', plate)
    print(' Bar: ', scn.bar)
    print(' Cup: ', scn.cup)
    print(' ====================== \n')

    return scn



if __name__ == '__main__':
    main()
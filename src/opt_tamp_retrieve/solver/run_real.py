# Description: Run the real world experiment
# rosfranka need to run in the background
# and run 'source /home/liu/catkin_ws/devel/setup.bash'


import os, sys, time
import pybullet as pb
import pandas as pd
import numpy as np
import math

from opt_solver import Solver
from execute_real import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
# from pddlstream.language.constants import Equal, And

from common.scn import Scenario 
from camera.camera import Camera

from examples.pybullet.utils.pybullet_tools.utils import connect
from examples.pybullet.utils.pybullet_tools.franka_primitives import Observe
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

RUN_TIMES = 1


def main():

    # Observe and Build the solving world

    sim_id = connect(use_gui=True) 
    scn = Scenario(bullet_client=sim_id)
    scn.reset()
    scn.add_robot() # 1
    scn.add_cup(((0.5, 0, 0.035), (0,0,0,1))) # 2
    scn.add_hook() # 3
    scn.add_goal_surface(((0.2, -0.3, 0.01), (0,0,0,1))) # 4

    print(f'panda = {scn.panda}')
    print(f'cup = {scn.cup}')
    print(f'hook = {scn.hook}')
    print(f'goal_surface = {scn.goal_surface}')


    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('Press enter to continue: ')
    #################################################
    # Dataframes
    df_pt = pd.DataFrame(columns=['planning_time'])
    df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])
    plan_success = 0
    execute_success = 0

    #################################################
    # Solve the problem
    solver = Solver(scn.panda, movable=[scn.cup, scn.hook], tools=[scn.hook], surfaces=[scn.goal_surface], marks=[])
    goal = ('on', scn.cup, scn.goal_surface)

    # executor = Executor(scn.panda, scn)
    executor = Executor(scn)
    # solver = Solver(scn.panda, movable=[cube1, cube2, scn.hook], tools=[scn.hook], surfaces=[scn.goal_surface, scn.hook_mark], marks=[scn.hook_mark])
    # goal = And(('on', cube1, scn.goal_surface), ('on', cube2, scn.goal_surface)) #, ('on', cube2, scn.goal_surface)

    
    for i in range(RUN_TIMES):

        print("\033[92m {}\033[00m" .format(f'\n {i} Episode'))

        reset(scn)

        # Plan
        start = time.time()
        plan, cost, evaluations = solver.solve(goal)
        print('Plan:', plan) 
        end = time.time()
        planning_time = end - start

        if plan is None:
            continue
        else:
            plan_success += 1

        # Execute
        # try:
        executor.execute(solver.problem, plan)
        execute_success += 1
        # except:
        #     print('Execution failed')
        #     continue

        # Record
        df_pt.loc[i] = [planning_time]
        df_su.loc[i] = [plan_success, execute_success]
        # df_pt.to_csv('./results/planning_time.csv', index=False)
        # df_su.to_csv('./results/success.csv', index=False)

    # input('Press enter to continue: ')

#################################################

def reset(scn):

    quat = quaternion_from_euler(math.pi, 0, 0)
    observe = Observe()
    # camera = Camera()
    pose_dict = observe.apply(expected_ids=['0', '1']) #camera.aruco_pose_detection(observe_pose)
    
    # Cup
    cup_id = '0'
    cup_pose = ((pose_dict[cup_id][0][0], pose_dict[cup_id][0][1], 0.05), pose_dict[cup_id][1])
    pb.resetBasePositionAndOrientation(scn.cup, cup_pose[0], cup_pose[1])
    
    # Hook
    hook_id = '1'
    hook_pose = pose_dict[hook_id]
    pb.resetBasePositionAndOrientation(scn.hook, hook_pose[0], hook_pose[1])

    # Goal
    goal_id = '5'
    # goal_pose = ((pose_dict[goal_id][0][0], pose_dict[goal_id][0][1], pose_dict[goal_id][0][2] + 0.02), pose_dict[goal_id][1])
    # pb.resetBasePositionAndOrientation(scn.goal_surface, goal_pose[0], goal_pose[1])
    # pb.resetBasePositionAndOrientation(scn.goal_surface, (0.1, 0.4, 0.05), (0,0,0,1))

    pb.setRealTimeSimulation(1)
    time.sleep(2)
    pb.setRealTimeSimulation(0)

    

if __name__ == '__main__':
    main()

import os, sys, time, math
import pybullet as pb
import pandas as pd
import numpy as np

from opt_solver import Solver
from execute_real import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../')
sys.path.insert(0, file_path + '/../../pddlstream/')

from examples.pybullet.utils.pybullet_tools.franka_primitives import Observe, ObserveObject
from examples.pybullet.utils.pybullet_tools.transformations import quaternion_from_euler, euler_from_quaternion

from common.scn import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect, create_obj
import pybullet as p
import rospy

RUN_TIMES = 1


def main():

    rospy.init_node('run_real', anonymous=True)
    sim_id = connect(use_gui=True) 
    scn = Scenario()
    scn.reset()
    scn.add_robot(pose=((0,0,0), (0,0,0,1))) # 1
    table1 = scn.add_low_table() # 2
    table2 = scn.add_high_table_real() # 3
    
    scn = reset(scn)
    
    p.changeDynamics(scn.dish, -1, lateralFriction=1)


    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)


    input('Press enter to continue: ')

    #################################################
    # Solve the problem
    solver = Solver(scn.panda, movable=[scn.dish], tools=[], surfaces=[table1, table2], marks=[])
    executor = Executor(scn)

    
    #################################################
    # Solve the problem
    df_pt = pd.DataFrame(columns=['planning_time'])
    df_su = pd.DataFrame(columns=['plan_success', 'execute_success'])

    for i in range(RUN_TIMES):
        print(f' \nEpisode {i} \n')
        goal = ('on', scn.dish, table2)
        start = time.time()
        plan, cost, evaluations = solver.solve(goal)
        end = time.time()
        planning_time = end - start
        df_pt.loc[i] = [planning_time]
        
        if plan is None:
            continue
        else:
            plan_success += 1
        #################################################

        input(' Press enter to execute: ')  
        executor.execute(solver.problem, plan)


def reset(scn):
    observe = Observe()
    pose_dict = observe.apply(expected_ids=['3'])
    dish_pose = pose_dict['3']

    observe_object = ObserveObject()
    observe_object.reconstruct(dish_pose)

    mark_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../camera/'
    pb.setAdditionalSearchPath(mark_dir)
    
    scn.dish = create_obj('./hull.obj', mass=1)
    quat = quaternion_from_euler(0, 0, -math.pi/2)
    pb.resetBasePositionAndOrientation(scn.dish, np.array(dish_pose[0])+ np.array([0,0,0.01]), quat)

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)

    return scn

    

if __name__ == '__main__':
    main()
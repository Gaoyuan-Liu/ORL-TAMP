
import os, sys, time
import pybullet as pb
import pandas as pd
import numpy as np
import math

from opt_solver import Solver
from post_process import Executor

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + '/../')
sys.path.insert(0, file_path + '/../../pddlstream/')
from pddlstream.language.constants import Equal, And

from scenario import Scenario

from examples.pybullet.utils.pybullet_tools.utils import connect
from policies_sb3 import HookingPolicy


RUN_TIMES = 50
def main():

    
    sim_id = connect(use_gui=True) 
    scn = Scenario()
    scn.reset()
    scn.add_robot()

    cup_1 = scn.add_cup(((0.9, -0.3, 0.035), (0,0,0,1)))
    cup_2 = scn.add_cup(((0.9, 0.3, 0.035), (0,0,0,1)))
    cups = [cup_1, cup_2]


    scn.add_hook()

    scn.add_goal_surface(((-0.4, 0.0, 0.01), (0,0,0,1)))

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
    # Planning
    solver = Solver(scn.panda, movable=scn.cups + [scn.hook], tools=[scn.hook], surfaces=[scn.goal_surface], marks=[])
    executor = Executor(scn.panda, scn)
    # solver = Solver(scn.panda, movable=[cube1, cube2, scn.hook], tools=[scn.hook], surfaces=[scn.goal_surface, scn.hook_mark], marks=[scn.hook_mark])
    # goal = And(('on', cube1, scn.goal_surface), ('on', cube2, scn.goal_surface)) #, ('on', cube2, scn.goal_surface)
    goal = And(('on', cup_1, scn.goal_surface), ('on', cup_2, scn.goal_surface))
    # goal = ('on', scn.hook, scn.hook_mark)
    
    # RL Helper
    hook = HookingPolicy(scn)
    

    for i in range(RUN_TIMES):
        print("\033[92m {}\033[00m" .format(f'\n {i} Episode'))

        reset(scn)
        n_solved = 0

        start = time.time()

        for cup in scn.cups:
            goal = ('on', cup, scn.goal_surface)
            while n_solved < 2:
                plan, cost, evaluations = solver.solve(goal)
                
                if plan is None: # If cannot solve, use RL
                    hook.specify(cup, None) # Object, Goal
                    hook.apply(None)
                else:
                    n_solved += 1
                    break
            
                # Timeout
                now = time.time()
                planning_time_in = now - start
                if planning_time_in > 50:
                    break

        end = time.time()
        planning_time = end - start 
        df_pt.loc[i] = [planning_time]       
        
        if n_solved == 2:
            plan_success += 1
            execute_success += 1

        # df_pt.loc[i] = [planning_time]    
        df_su.loc[i] = [plan_success, execute_success]

        df_pt.to_csv('./results/planning_time.csv', index=False)
        df_su.to_csv('./results/success.csv', index=False)



    #################################################
    # Execute the plan
    # if plan is None:
    #     exit()
    # print('Plan:', plan)    
    
    # executor.execute(solver.problem, plan, scn)


    # input('Press enter to continue: ')
def reset(scn):
    # Cups
    for i in scn.cups:
            r = np.random.uniform(low=0.8, high=1.)
            theta = np.random.uniform(low=-math.pi/4, high=math.pi/4)
            x = r*math.cos(theta)
            y = r*math.sin(theta)
            z = 0.025

            # yaw = np.random.uniform(low=-math.pi, high=math.pi)
            orn = [0, 0, 0, 1] #quaternion_from_euler(0,0,yaw)
            pb.resetBasePositionAndOrientation(i, [x, y, z], orn)

    # Hook
    pb.resetBasePositionAndOrientation(scn.hook, (0.55,0,0.1), (0,0,0,1))

    pb.setRealTimeSimulation(1)
    time.sleep(1)
    pb.setRealTimeSimulation(0)    


if __name__ == '__main__':
    main()